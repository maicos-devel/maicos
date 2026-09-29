# Copyright (c) 2026 Authors and contributors
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

logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricCylinder(CylinderBase):
    r"""Cylindrical dielectric profiles.

    Computes the axial :math:`\varepsilon_z(r)`, azimuthal :math:`\varepsilon_{\phi}(r)`
    and inverse radial :math:`\varepsilon_r^{-1}(r)` components of the cylindrical
    dielectric tensor :math:`\varepsilon`.
    The components are binned along the radial direction of the
    cylinder. The :math:`z`-axis of the cylinder is pointing in the direction given by
    the ``dim`` parameter. The center of the cylinder is either located at the center of
    the simulation box (default) or at the center of mass of the ``refgroup``, if
    provided.

    For usage please refer to the
    :ref:`sphx_glr_generated_examples_dielectrics_dielectric-profiles.py` example and
    for details on the theory see :ref:`dielectric-explanations`.

    For correlation analysis, the component along the :math:`z`-axis is used.
    ${CORRELATION_INFO}

    Also, please read and cite :footcite:p:`locheGiantaxialDielectric2019`.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${TEMPERATURE_PARAMETER}
    vcutwidth : float
        Spacing of virtual cuts (bins) along the parallel directions.
    single : bool
        For a single chain of molecules the average of :math:`M` is zero. This flag sets
        :math:`\langle M \rangle = 0`.
    ${CYLINDER_CLASS_PARAMETERS}
    ${BASE_CLASS_PARAMETERS}
    ${OUTPUT_PREFIX_PARAMETER}

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
    results.eps_phi : numpy.ndarray
        Reduced azimuthal dielectric profile
        :math:`(\varepsilon_{\phi}(r) - 1)`
    results.deps_phi : numpy.ndarray
        Estimated uncertainty of azimuthal dielectric profile.


    """

    def __init__(
        self,
        atomgroup: mda.AtomGroup,
        temperature: float = 300,
        vcutwidth: float = 0.1,
        single: bool = False,
        dim: int = 2,
        zmin: float | None = None,
        zmax: float | None = None,
        rmin: float = 0,
        rmax: float | None = None,
        bin_width: float = 0.1,
        refgroup: mda.AtomGroup | None = None,
        unwrap: bool = True,
        pack: bool = True,
        jitter: float = 0.0,
        concfreq: int = 0,
        output_prefix: str = "eps_cyl",
    ) -> None:
        self._locals = locals()
        self.comp = get_compound(atomgroup)
        ix = atomgroup._get_compound_indices(self.comp)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

        if zmin is not None or zmax is not None or rmin != 0 or rmax is not None:
            logger.warning(
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
            "Analysis of the axial, azimuthal and inverse radial "
            "components of the cylindrical dielectric tensor."
        )
        # Print Philip Loche citation
        logger.info(citation_reminder("10.1021/acs.jpcb.9b09269"))

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

        # We don't need to to know m_r for the whole system, since we calculate M_r
        # directly from the atom positions and charges. But maybe someone wants to use
        # it for something else, so we calculate it here as well.
        self._obs.m_r_tot = (
            -np.cumsum(curQ_r_tot) / 2 / np.pi / self._obs.L / self._obs.bin_pos
        )

        # Direct calculation without binning of
        # \int_0^R dr m(r) = -1/(2πL) * sum_i q_i log(R/r_i)
        r_atoms = self.pos_cyl[:, 0]
        # Set r=0 to the smallest non-zero bin edge to avoid numerical errors. Atoms at
        # r=0 should be very rare in practice.
        r_atoms[r_atoms == 0.0] = 0.001
        self._obs.M_r = -np.sum(
            self._universe.atoms.charges * np.log(self._obs.bin_edges[-1] / r_atoms)
        ) / (2 * np.pi * self._obs.L)

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
        self.phicutwidth = self.vcutwidth  # angstrom

        nbinsphi = np.ceil(self.rmax * 2 * np.pi / self.phicutwidth).astype(int)
        nbinsphi = nbinsphi // 4 * 4
        logging.debug("Number of phi bins {nbinsphi}")

        rbins, testrpos = self._get_coc_rbins()

        # ------------ FIRST CUT --------------------
        # do the first cut at the 0, 2pi plane.
        # Wrap all molecules that cross the 0, 2pi plane to their center of charge.
        # Thus excess charges across virtual cuts are only from molecules at the cut.
        # And not from molecules across the first cut (the 0, 2pi) plane.
        # Note: Instead of wrapping we could also delete them from the atomgroup.
        pos_phi = self._wrap_phi_positions(testrpos, shift=0)

        # put the atoms in phi bins and calculate the charge in each phi bin
        phi = (np.arange(nbinsphi)) * (2 * np.pi / nbinsphi)
        phibins = np.digitize(pos_phi, phi[1:])

        # calculate the charge in each bin
        curQphi = np.bincount(
            rbins + self.n_bins * phibins,
            weights=self.atomgroup.charges,
            minlength=self.n_bins * nbinsphi,
        ).reshape(nbinsphi, self.n_bins)
        # print(curQphi)

        # calculate the integral over the charge density
        area = self._obs.bin_width * self._obs.L
        curqphi = np.cumsum(curQphi, axis=0) / (area)
        # print("first cutting")
        # print(curqphi)

        # ------------- SECOND CUTTING --------------

        # cut along at pi
        pos_phi = self._wrap_phi_positions(testrpos, shift=np.pi)

        # put the atoms in phi bins and calculate the charge in each phi bin
        phibins = np.digitize(pos_phi, phi[1:])

        # calcualte the charge in each bin
        curQphi = np.bincount(
            rbins + self.n_bins * phibins,
            weights=self.atomgroup.charges,
            minlength=self.n_bins * nbinsphi,
        ).reshape(nbinsphi, self.n_bins)

        # calculate the integral over the charge density
        area = self._obs.bin_width * self._obs.L
        curqphi2 = np.cumsum(curQphi, axis=0) / (area)
        # print("second cutting")
        # print(curqphi2)

        # puzzle the two halfs together
        whole = curqphi.shape[0]
        quat = whole // 4
        half = quat * 2
        tquat = quat * 3

        curqphi[:quat, :] = curqphi2[half:tquat, :]
        curqphi[tquat:, :] = curqphi2[quat:half, :]

        # print("puzzled together")
        # print(curqphi)

        # average of m_phi(phi, r) over phi
        self._obs.m_phi = -curqphi.mean(axis=0)
        # This is the systems dipole moment in phi_direction and
        # not the radial integral of the dipole density.
        # also it's already multiplied by 2 pi L
        self._obs.M_phi = np.sum(
            self._obs.bin_volume * self._obs.bin_pos * self._obs.m_phi
        )
        self._obs.mM_phi = self._obs.m_phi * self._obs.M_phi

        # Save the total dipole moment in z dierection for correlation analysis.
        return self._obs.M_z

    def _get_coc_rbins(self):
        """Move all r-positions to 'center of absolute charge'.

        Thus we avoid monopoles in r-direction.
        We only want to cut in phi and z direction.
        """
        chargepos = self.pos_cyl[self.atomgroup.ix, 0] * np.abs(self.atomgroup.charges)
        center = self.atomgroup.accumulate(
            chargepos, compound=self.comp
        ) / self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges), compound=self.comp
        )
        testpos = center[self.inverse_ix]

        rbins = np.digitize(testpos, self._obs.bin_edges[1:-1])
        return rbins, testpos

    def _coc_phi_shifted(self, shift):
        """Calculate the center of charge in phi direction."""
        pos_phi = self.pos_cyl[self.atomgroup.ix, 1] + shift
        pos_phi = np.mod(pos_phi, 2 * np.pi)
        chargepos_phi = pos_phi * np.abs(self.atomgroup.charges)
        center_phi = self.atomgroup.accumulate(
            chargepos_phi, compound=self.comp
        ) / self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges), compound=self.comp
        )
        # expand center_phi to be on a per atom basis
        test_center_phi = center_phi[self.inverse_ix]
        return test_center_phi, pos_phi

    def _wrap_phi_positions(self, testrpos, shift=0):
        """Wrap molecules that cross 0 to 2pi plane to their center of charge."""
        test_center_phi, pos_phi = self._coc_phi_shifted(shift)

        # wrap all components that cross the border at 0 and 2 pi
        difference = test_center_phi - pos_phi
        # check if there are atoms within the molecule with a
        # distance greater than 1 angstrom (arclength)
        difference_in_atom = np.where(np.abs(difference) * testrpos > 1, 1, 0)
        difference_in_molecule = self.atomgroup.accumulate(
            difference_in_atom, compound=self.comp
        )
        is_diff = difference_in_molecule[self.inverse_ix]
        return np.where(is_diff, test_center_phi, pos_phi)

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
            cov_phi = self.means.mM_phi - self.means.m_phi * self.means.M_phi

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

            dcov_phi = np.sqrt(
                self.sems.mM_phi**2
                + self.sems.m_phi**2 * self.means.M_phi**2
                + self.means.m_phi**2 * self.sems.M_phi**2
            )
        else:
            # <M> = 0 for a single line of water molecules.
            cov_z = self.means.mM_z
            cov_r = self.means.mM_r
            cov_phi = self.means.mM_phi
            dcov_z = self.sems.mM_z
            dcov_r = self.sems.mM_r
            dcov_phi = self.sems.mM_phi

        self.results.m_z = self.means.m_z
        self.results.dm_z = self.sems.m_z
        self.results.eps_z = self._pref * cov_z
        self.results.deps_z = self._pref * dcov_z

        self.results.m_r = self.means.m_r
        self.results.dm_r = self.sems.m_r
        self.results.eps_r = -(
            2 * np.pi * self._obs.L * self._pref * self.results.bin_pos * cov_r
        )
        self.results.deps_r = (
            2 * np.pi * self._obs.L * self._pref * self.results.bin_pos * dcov_r
        )

        self.results.m_phi = self.means.m_phi
        self.results.dm_phi = self.sems.m_phi

        self.results.eps_phi = 1 / self.results.bin_pos * self._pref * cov_phi
        self.results.deps_phi = 1 / self.results.bin_pos * self._pref * dcov_phi

    @render_docs
    def save(self) -> None:
        """${SAVE_METHOD_PREFIX_DESCRIPTION}"""  # noqa: D415
        outdata_z = np.array(
            [
                self.results.bin_pos,
                self.results.eps_z,
                self.results.deps_z,
            ]
        ).T
        outdata_r = np.array(
            [
                self.results.bin_pos,
                self.results.eps_r,
                self.results.deps_r,
            ]
        ).T

        outdata_phi = np.array(
            [
                self.results.bin_pos,
                self.results.eps_phi,
                self.results.deps_phi,
            ]
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

        columns += ["ε_phi - 1", "Δε_phi"]

        self.savetxt(
            "{}{}".format(self.output_prefix, "_phi.dat"), outdata_phi, columns=columns
        )
