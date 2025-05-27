#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Module for computing cylindrical dielectric profiles."""

import logging

import MDAnalysis as mda
import numpy as np
import scipy.constants

from ..core import CylinderBase
from ..lib.util import charge_neutral, citation_reminder, get_compound, render_docs


@render_docs
@charge_neutral(filter="error")
class DielectricCylinder(CylinderBase):
    r"""Cylindrical dielectric profiles.

    Computes the axial :math:`\varepsilon_z(r)` and inverse radial
    :math:`\varepsilon_r^{-1}(r)` components of the cylindrical dielectric tensor
    :math:`\varepsilon`. The components are binned along the radial direction of the
    cylinder. The :math:`z`-axis of the cylinder is pointing in the direction given by
    the ``dim`` parameter. The center of the cylinder is either located at the center of
    the simulation box (default) or at the center of mass of the ``refgroup``, if
    provided.

    For usage please refer to :ref:`How-to: Dielectric constant<howto-dielectric>` and
    for details on the theory see :ref:`dielectric-explanations`.

    For correlation analysis, the component along the :math:`z`-axis is used.
    ${CORRELATION_INFO}

    Also, please read and cite :footcite:p:`locheGiantaxialDielectric2019`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    ${TEMPERATURE_PARAMETER}
    single : bool
        For a single chain of molecules the average of :math:`M` is zero. This flag sets
        :math:`\langle M \rangle = 0`.

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.eps_z : numpy.ndarray
        Reduced axial dielectric profile :math:`(\varepsilon_z(r) - 1)` of the
        selected atomgroup
    results.deps_z : numpy.ndarray
        Estimated uncertainty of axial dielectric profile
    results.eps_r : numpy.ndarray
        Reduced inverse radial dielectric profile
        :math:`(\varepsilon^{-1}_r(r) - 1)`
    results.deps_r : numpy.ndarray
        Estimated uncertainty of inverse radial dielectric profile

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        bin_width: float = 0.1,
        temperature: float = 300,
        single: bool = False,
        output_prefix: str = "eps_cyl",
        refgroup: mda.AtomGroup | None = None,
        concfreq: int = 0,
        jitter: float = 0.0,
        dim: int = 2,
        rmin: float = 0,
        rmax: float | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        vcutwidth: float = 0.1,
        unwrap: bool = True,
        pack: bool = True,
    ) -> None:
        self._locals = locals()
        self.comp = get_compound(atomgroup)
        ix = atomgroup._get_compound_indices(self.comp)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

        if zmin is not None or zmax is not None or rmin != 0 or rmax is not None:
            logging.warning(
                "Setting `rmin` and `rmax` (as well as `zmin` and `zmax`) might cut "
                "off molecules. This will lead to severe artifacts in the dielectric "
                "profiles."
            )

        super().__init__(
            atomgroup,
            unwrap=unwrap,
            pack=pack,
            refgroup=refgroup,
            jitter=jitter,
            concfreq=concfreq,
            dim=dim,
            zmin=zmin,
            zmax=zmax,
            bin_width=bin_width,
            rmin=rmin,
            rmax=rmax,
            wrap_compound=self.comp,
        )
        self.output_prefix = output_prefix
        self.temperature = temperature
        self.single = single
        self.vcutwidth = vcutwidth

    def _prepare(self) -> None:
        logging.info(
            "Analysis of the axial and inverse radial "
            "components of the cylindrical dielectric tensor."
        )
        # Print Philip Loche citation
        logging.info(citation_reminder("10.1021/acs.jpcb.9b09269"))

        super()._prepare()

    def _single_frame(self) -> float:
        super()._single_frame()

        # Precalculate the bins each atom belongs to.
        rbins = np.digitize(self.pos_cyl[:, 0], self._obs.bin_edges[1:-1])

        # Calculate the charge per bin for the selected atomgroup.
        curQ_r = np.bincount(
            rbins[self.atomgroup.ix],
            weights=self.atomgroup.charges,
            minlength=self.n_bins,
        )

        # In literature, the charge density is integrated along the radial direction to
        # get the dipole moment density. We can rewrite the integral by identifying:
        # q(a) = 2π * L * int_0^a ρ(r) * r dr,
        # where q(a) is the charge enclosed within a cylinder of radius a and length L.
        # This allows us to avoid numerical errors.
        self._obs.m_r = -np.cumsum(curQ_r) / 2 / np.pi / self._obs.L / self._obs.bin_pos

        # Same as above, but for the whole system.
        curQ_r_tot = np.bincount(
            rbins, weights=self._universe.atoms.charges, minlength=self.n_bins
        )

        self._obs.m_r_tot = (
            -np.cumsum(curQ_r_tot) / 2 / np.pi / self._obs.L / self._obs.bin_pos
        )

        # Note that M_r is not really the total system dipole moment in radial
        # direction, but it keeps the nomenclature consistent across all of the
        # dielectric modules.
        self._obs.M_r = np.sum(self._obs.m_r_tot * self._obs.bin_width)
        self._obs.mM_r = self._obs.m_r * self._obs.M_r

        # Use virtual cutting method (for axial component)
        # ========================================================
        # number of virtual cuts ("many")
        nbinsz = np.ceil(self._obs.L / self.vcutwidth).astype(int)

        # Move all r-positions to 'center of charge' such that we avoid monopoles in
        # r-direction. We only want to cut in z direction.
        chargepos = self.pos_cyl[self.atomgroup.ix, 0] * np.abs(self.atomgroup.charges)
        center = self.atomgroup.accumulate(
            chargepos, compound=self.comp
        ) / self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges), compound=self.comp
        )
        testpos = center[self.inverse_ix]

        rbins = np.digitize(testpos, self._obs.bin_edges[1:-1])
        z = (np.arange(nbinsz)) * (self._obs.L / nbinsz)
        zbins = np.digitize(self.pos_cyl[self.atomgroup.ix, 2], z[1:])

        curQz = np.bincount(
            rbins + self.n_bins * zbins,
            weights=self.atomgroup.charges,
            minlength=self.n_bins * nbinsz,
        ).reshape(nbinsz, self.n_bins)

        curqz = np.cumsum(curQz, axis=0) / (self._obs.bin_area)[np.newaxis, :]
        self._obs.m_z = -curqz.mean(axis=0)
        # This is the systems dipole moment in z-direction and
        # not the radial integral of the dipole density.
        self._obs.M_z = np.dot(self._universe.atoms.charges, self.pos_cyl[:, 2])
        self._obs.mM_z = self._obs.m_z * self._obs.M_z

        # Use virtual cutting method (for azimuthal component)
        # ========================================================
        # number of virtual cuts ("many")
        self.phicutwidth = 0.1 # angstrom

        nbinsphi = np.ceil(self.rmax * 2 * np.pi / self.phicutwidth).astype(int)
        logging.debug("Number of phi bins {nbinsphi}")

        # Move all r-positions to 'center of charge' such that we avoid monopoles in
        # r-direction. We only want to cut in z direction.
        chargepos = self.pos_cyl[self.atomgroup.ix, 0] * np.abs(self.atomgroup.charges)
        center = self.atomgroup.accumulate(
            chargepos, compound=self.comp
        ) / self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges), compound=self.comp
        )
        testpos = center[self.inverse_ix]

        # do the first cut at the 0, 2pi plane.
        # Wrap all molecules that cross the 0, 2pi plane to their center of charge
        # tbh, we can just delete them from the atomgroup
        chargepos_phi = self.pos_cyl[self.atomgroup.ix, 1] * np.abs(self.atomgroup.charges)
        center_phi = self.atomgroup.accumulate(
            chargepos_phi, compound=self.comp
        ) / self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges), compound=self.comp
        )
        # expand center_phi to be on a per atom basis
        test_center_phi = chargepos_phi[self.inverse_ix]
        # wrap all components that cross the border
        pos_phi = self.pos_cyl[self.atomgroup.ix, 1]
        difference = test_center_phi - pos_phi
        pos_phi = np.where(difference > 1.9*np.pi, pos_phi + 2 * np.pi, pos_phi)
        pos_phi = np.where(difference < -1.9*np.pi, pos_phi -2*np.pi, pos_phi)


        rbins = np.digitize(testpos, self._obs.bin_edges[1:-1])
        phi = (np.arange(nbinsphi)) * (2 * np.pi / nbinsphi)
        phibins = np.digitize(pos_phi, phi[1:])

        curQphi = np.bincount(
            rbins + self.n_bins * phibins,
            weights=self.atomgroup.charges,
            minlength=self.n_bins * nbinsphi,
        ).reshape(nbinsphi, self.n_bins)

        area = self._obs.bin_width * self._obs.L
        curqphi = np.cumsum(curQphi, axis=0) / (area )
        self._obs.m_phi = -curqphi.mean(axis=0)
        # This is the systems dipole moment in z-direction and
        # not the radial integral of the dipole density.
        self._obs.M_phi = np.sum(self._obs.bin_pos**2 * self._obs.m_phi)
        self._obs.mM_phi = self._obs.m_phi * self._obs.M_phi

        # Save the total dipole moment in z dierection for correlation analysis.
        return self._obs.M_z

    def _conclude(self) -> None:
        super()._conclude()

        self._pref = 1 / scipy.constants.epsilon_0
        self._pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        self._pref /= (
            scipy.constants.angstrom / (scipy.constants.elementary_charge) ** 2
        )

        if not self.single:
            # A factor of 2 pi L cancels out in the final expression because here M_z is
            # the total dipole moment in z-direction, not the radial integral of the
            # dipole density. M_z = 2 pi L \int_0^R dr r m(r)
            cov_z = self.means.mM_z - self.means.m_z * self.means.M_z
            cov_r = self.means.mM_r - self.means.m_r * self.means.M_r

            dcov_z = np.sqrt(
                self.sems.mM_z**2
                + self.sems.m_z**2 * self.means.M_z**2
                + self.means.m_z**2 * self.sems.M_z**2
            )
            dcov_r = np.sqrt(
                self.sems.mM_r**2
                + self.sems.m_r**2 * self.means.M_r**2
                + self.means.m_r**2 * self.sems.M_r**2
            )
        else:
            # <M> = 0 for a single line of water molecules.
            cov_z = self.means.mM_z
            cov_r = self.means.mM_r
            dcov_z = self.sems.mM_z
            dcov_r = self.sems.mM_r

        self.results.eps_z = self._pref * cov_z
        self.results.deps_z = self._pref * dcov_z

        self.results.eps_r = -(
            2 * np.pi * self._obs.L * self._pref * self.results.bin_pos * cov_r
        )
        self.results.deps_r = (
            2 * np.pi * self._obs.L * self._pref * self.results.bin_pos * dcov_r
        )

        self.results.m_phi = self.means.m_phi
        cov_phi = self.means.mM_phi - self.means.m_phi * self.means.M_phi
        self.results.eps_phi = 1 + 2*np.pi *self._obs.L / self.results.bin_pos * self._pref * cov_phi  

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_DESCRIPTION}"""  # noqa: D415
        outdata_z = np.array(
            [self.results.bin_pos, self.results.eps_z, self.results.deps_z]
        ).T
        outdata_r = np.array(
            [self.results.bin_pos, self.results.eps_r, self.results.deps_r]
        ).T

        outdata_phi = np.array(
                [self.results.bin_pos, self.results.eps_phi, self.results.m_phi]
                ).T

        columns = ["positions [Å]"]

        columns += ["ε_z - 1", "Δε_z"]

        self.savetxt(
            "{}{}".format(self.output_prefix, "_z.dat"), outdata_z, columns=columns
        )

        columns = ["positions [Å]"]

        columns += ["ε^-1_r - 1", "Δε^-1_r"]

        self.savetxt(
            "{}{}".format(self.output_prefix, "_r.dat"), outdata_r, columns=columns
        )

        columns = ["positions [Å]"]

        columns += ["ε_phi", "m_phi"]

        self.savetxt(
            "{}{}".format(self.output_prefix, "_phi.dat"), outdata_phi, columns=columns
        )


