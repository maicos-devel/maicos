#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Tools for computing relative permittivities.

The dielectric constant modules of MAICoS allow for computing dielectric
profile and dielectric spectrum from molecular simulation trajectory files.
"""

import logging

import numpy as np
import scipy.constants

from ..core import AnalysisBase, CylinderBase, PlanarBase, SphereBase
from ..lib.math import FT, iFT, symmetrize
from ..lib.util import bin, charge_neutral, get_compound, render_docs


logger = logging.getLogger(__name__)


@render_docs
@charge_neutral(filter="error")
class DielectricPlanar(PlanarBase):
    """Calculate planar dielectric profiles.

    See Schlaich, et al., Phys. Rev. Lett., vol. 117 (2016) for details.

    Parameters
    ----------
    ${ATOMGROUPS_PARAMETER}
    ${PLANAR_CLASS_PARAMETERS}
    xy : bool
        Use 2D slab geometry.
    vac : bool
        Use vacuum boundary conditions instead of metallic (2D only!).
    sym : bool
        Symmetrize the profiles.
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output files.
    vcutwidth : float
        Spacing of virtual cuts (bins) along the parallel directions.

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.dens_mean : numpy.ndarray
        eps_par: Parallel dielectric profile ε_∥
    results.deps_par : numpy.ndarray
        Error of parallel dielectric profile
    results.eps_par_self : numpy.ndarray
        Reduced self contribution of parallel dielectric profile (ε_∥_self - 1)
    results.eps_par_coll : numpy.ndarray
        Reduced collective contribution of parallel dielectric profile
        (ε_∥_coll - 1)
    results.eps_perp : numpy.ndarray
        Inverse perpendicular dielectric profile ε^{-1}_⟂
    results.deps_perp : numpy.ndarray
        Error of inverse perpendicular dielectric profile
    results.eps_perp_self : numpy.ndarray
        Reduced self contribution of Inverse perpendicular dielectric profile
        (ε^{-1}_⟂_self - 1)
    results.eps_perp_coll : numpy.ndarray
        Reduced collective contribution of Inverse perpendicular
        dielectric profile (ε^{-1}_⟂_coll - 1)
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=None,
                 zmax=None,
                 bin_width=0.5,
                 refgroup=None,
                 xy=False,
                 sym=False,
                 vac=False,
                 unwrap=True,
                 temperature=300,
                 output_prefix="eps",
                 concfreq=0,
                 vcutwidth=0.1):
        super(DielectricPlanar, self).__init__(atomgroups=atomgroups,
                                               dim=dim,
                                               zmin=zmin,
                                               zmax=zmax,
                                               bin_width=bin_width,
                                               refgroup=refgroup,
                                               unwrap=unwrap,
                                               multi_group=True)
        self.xy = xy
        self.sym = sym
        self.vac = vac

        self.temperature = temperature
        self.output_prefix = output_prefix
        self.concfreq = concfreq
        self.vcutwidth = vcutwidth

    def _prepare(self):
        super(DielectricPlanar, self)._prepare()

        self._obs.M_par = np.zeros(2)
        self._obs.M_perp = 0
        self._obs.M_perp_2 = 0

        n_ag = self.n_atomgroups

        self._obs.m_par = np.zeros((self.n_bins, 2, n_ag))
        self._obs.mM_par = np.zeros((self.n_bins, n_ag))
        self._obs.mm_par = np.zeros((self.n_bins, n_ag))
        self._obs.cmM_par = np.zeros((self.n_bins, n_ag))
        self._obs.cM_par = np.zeros((self.n_bins, 2, n_ag))

        self._obs.m_perp = np.zeros((self.n_bins, n_ag))
        self._obs.mM_perp = np.zeros((self.n_bins, n_ag))
        self._obs.mm_perp = np.zeros((self.n_bins, n_ag))
        self._obs.cmM_perp = np.zeros((self.n_bins, n_ag))
        self._obs.cM_perp = np.zeros((self.n_bins, n_ag))

        self.comp = []
        self.inverse_ix = []

        for sel in self.atomgroups:
            comp, ix = get_compound(sel.atoms, return_index=True)
            _, inverse_ix = np.unique(ix, return_inverse=True)
            self.comp.append(comp)
            self.inverse_ix.append(inverse_ix)

    def _single_frame(self):
        super(DielectricPlanar, self)._single_frame()

        # precalculate total polarization of the box
        self._obs.M = np.dot(self._universe.atoms.charges,
                             self._universe.atoms.positions)

        self._obs.M_perp = self._obs.M[self.dim]
        self._obs.M_perp_2 = self._obs.M[self.dim]**2
        self._obs.M_par = self._obs.M[self.odims]

        # Use polarization density (for perpendicular component)
        # ======================================================
        for i, sel in enumerate(self.atomgroups):
            zpos = np.zeros(len(sel))
            np.clip(sel.atoms.positions[:, self.dim],
                    self.zmin, self.zmax, zpos)

            curQ = np.histogram(zpos,
                                bins=self.n_bins,
                                range=[self.zmin, self.zmax],
                                weights=sel.atoms.charges)[0]

            self._obs.m_perp[:, i] = -np.cumsum(curQ / self._obs.bin_area)
            self._obs.mM_perp[:, i] = \
                self._obs.m_perp[:, i] * self._obs.M_perp
            self._obs.mm_perp[:, i] = \
                self._obs.m_perp[:, i]**2 * self._obs.bin_volume
            self._obs.cmM_perp[:, i] = self._obs.m_perp[:, i] \
                * (self._obs.M_perp
                    - self._obs.m_perp[:, i] * self._obs.bin_volume)

            self._obs.cM_perp[:, i] = self._obs.M_perp - \
                self._obs.m_perp[:, i] * self._obs.bin_volume

            # Use virtual cutting method (for parallel component)
            # ===================================================
            # Move all z-positions to 'center of charge' such
            # that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only
            # want to cut in x/y direction)
            testpos = \
                sel.center(weights=np.abs(sel.charges),
                           compound=self.comp[i])[self.inverse_ix[i], self.dim]

            # Average parallel directions
            for j, direction in enumerate(self.odims):
                # At this point we should not use the wrap, which causes
                # unphysical dipoles at the borders
                Lx = self._ts.dimensions[direction]
                Ax = self._ts.dimensions[self.odims[1 - j]] \
                    * self._obs.bin_width
                vbinsx = np.ceil(Lx / self.vcutwidth).astype(int)
                xpos = np.clip(sel.atoms.positions[:, direction], 0, Lx)

                curQx = np.histogram2d(
                    xpos, testpos,
                    bins=[vbinsx, self.n_bins],
                    range=[[0, Lx], [self.zmin, self.zmax]],
                    weights=sel.atoms.charges)[0]

                # integral over x, so uniself._ts of area
                self._obs.m_par[:, j, i] = \
                    -np.cumsum(curQx / Ax, axis=0).mean(axis=0)

            # Can not use array for operations below,
            # without extensive reshaping of each array...
            # Therefore, take first element only since the volume of each bin
            # is the same in planar geometry.
            bin_volume = self._obs.bin_volume[0]

            self._obs.mM_par[:, i] = \
                np.dot(self._obs.m_par[:, :, i],
                       self._obs.M_par)
            self._obs.mm_par[:, i] = (
                self._obs.m_par[:, :, i]
                * self._obs.m_par[:, :, i]
                ).sum(axis=1) \
                * bin_volume
            self._obs.cmM_par[:, i] = \
                (self._obs.m_par[:, :, i]
                 * (self._obs.M_par
                    - self._obs.m_par[:, :, i]
                    * bin_volume)
                 ).sum(axis=1)
            self._obs.cM_par[:, :, i] = \
                self._obs.M_par \
                - self._obs.m_par[:, :, i] \
                * bin_volume

        return self._obs.M_par[0]

    def _conclude(self):
        super(DielectricPlanar, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        self.results.pref = pref
        self.results.V = self.means.bin_volume.sum()

        # Perpendicular component
        # =======================
        cov_perp = self.means.mM_perp \
            - self.means.m_perp \
            * self.means.M_perp

        # Using propagation of uncertainties
        dcov_perp = np.sqrt(
            self.sems.mM_perp**2
            + (self.means.M_perp * self.sems.m_perp)**2
            + (self.means.m_perp * self.sems.M_perp)**2
            )

        var_perp = self.means.M_perp_2 - self.means.M_perp**2

        cov_perp_self = self.means.mm_perp \
            - (self.means.m_perp**2 * self.means.bin_volume[0])
        cov_perp_coll = self.means.cmM_perp \
            - self.means.m_perp * self.means.cM_perp

        if self.xy:
            self.results.eps_perp = -pref * cov_perp
            self.results.eps_perp_self = - pref * cov_perp_self
            self.results.eps_perp_coll = - pref * cov_perp_coll
            self.results.deps_perp = pref * dcov_perp
            if (self.vac):
                self.results.eps_perp *= 2. / 3.
                self.results.eps_perp_self *= 2. / 3.
                self.results.eps_perp_coll *= 2. / 3.
                self.results.deps_perp *= 2. / 3.

        else:
            self.results.eps_perp = \
                - cov_perp / (pref**-1 + var_perp / self.results.V)
            self.results.deps_perp = pref * dcov_perp

            self.results.eps_perp_self = \
                (- pref * cov_perp_self) \
                / (1 + pref / self.results.V * var_perp)
            self.results.eps_perp_coll = \
                (- pref * cov_perp_coll) \
                / (1 + pref / self.results.V * var_perp)

        self.results.eps_perp += 1

        # Parallel component
        # ==================
        cov_par = np.zeros((self.n_bins, self.n_atomgroups))
        dcov_par = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_self = np.zeros((self.n_bins, self.n_atomgroups))
        cov_par_coll = np.zeros((self.n_bins, self.n_atomgroups))

        for i in range(self.n_atomgroups):
            cov_par[:, i] = 0.5 * (self.means.mM_par[:, i]
                                   - np.dot(self.means.m_par[:, :, i],
                                            self.means.M_par))

            # Using propagation of uncertainties
            dcov_par[:, i] = 0.5 * np.sqrt(
                self.sems.mM_par[:, i]**2
                + np.dot(self.sems.m_par[:, :, i]**2,
                         self.means.M_par**2)
                + np.dot(self.means.m_par[:, :, i]**2,
                         self.sems.M_par**2)
                )

            cov_par_self[:, i] = 0.5 * (
                self.means.mm_par[:, i]
                - np.dot(self.means.m_par[:, :, i],
                         self.means.m_par[:, :, i].sum(axis=0)))
            cov_par_coll[:, i] = \
                0.5 * (self.means.cmM_par[:, i]
                       - (self.means.m_par[:, :, i]
                       * self.means.cM_par[:, :, i]).sum(axis=1))

        self.results.eps_par = pref * cov_par
        self.results.deps_par = pref * dcov_par
        self.results.eps_par_self = pref * cov_par_self
        self.results.eps_par_coll = pref * cov_par_coll

        self.results.eps_par += 1

        if self.sym:
            symmetrize(self.results.eps_perp, axis=0, inplace=True)
            symmetrize(self.results.deps_perp, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_self, axis=0, inplace=True)
            symmetrize(self.results.eps_perp_coll, axis=0, inplace=True)

            symmetrize(self.results.eps_par, axis=0, inplace=True)
            symmetrize(self.results.deps_par, axis=0, inplace=True)
            symmetrize(self.results.eps_par_self, axis=0, inplace=True)
            symmetrize(self.results.eps_par_coll, axis=0, inplace=True)

    def save(self):
        """Save results."""
        outdata_perp = np.hstack([
            self.results.bin_pos[:, np.newaxis],
            self.results.eps_perp.sum(axis=1)[:, np.newaxis],
            np.linalg.norm(self.results.deps_perp, axis=1)[:, np.newaxis],
            self.results.eps_perp,
            self.results.deps_perp,
            self.results.eps_perp_self.sum(axis=1)[:, np.newaxis],
            self.results.eps_perp_coll.sum(axis=1)[:, np.newaxis],
            self.results.eps_perp_self,
            self.results.eps_perp_coll
            ])
        outdata_par = np.hstack([
            self.results.bin_pos[:, np.newaxis],
            self.results.eps_par.sum(axis=1)[:, np.newaxis],
            np.linalg.norm(self.results.deps_par, axis=1)[:, np.newaxis],
            self.results.eps_par,
            self.results.deps_par,
            self.results.eps_par_self.sum(axis=1)[:, np.newaxis],
            self.results.eps_par_coll.sum(axis=1)[:, np.newaxis],
            self.results.eps_par_self,
            self.results.eps_par_coll
            ])

        columns = ["position [Å]", "ε_r (system)", "Δε_r (system)"]
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"ε_r ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"Δε_r ({i+1})")
        columns += ["self ε_r - 1 (system)", "coll. ε_r - 1 (system)"]
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"self ε_r - 1 ({i+1})")
        for i, _ in enumerate(self.atomgroups):
            columns.append(f"coll. ε_r - 1 ({i+1})")

        self.savetxt("{}{}".format(self.output_prefix, "_perp"),
                     outdata_perp, columns=columns)
        self.savetxt("{}{}".format(self.output_prefix, "_par"),
                     outdata_par, columns=columns)


@render_docs
@charge_neutral(filter="error")
class DielectricCylinder(CylinderBase):
    """Calculate cylindrical dielectric profiles.

    Components are calculated along the axial (z) and radial (along xy)
    direction at the system's center of mass.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${CYLINDER_CLASS_PARAMETERS}
    atomgroup : AtomGroup
        :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the dielectric profiles are calculated
    geometry : str
        A structure file without water from which com is calculated.
    radius : float
        Radius of the cylinder (Å)
    bin_width : float
        Bindiwdth the bin_width (Å)
    variable_dr : bool
        Use a variable bin_width, where the volume is kept fixed.
    length : float
        Length of the cylinder (Å)
    temperature : float
        temperature (K)
    single : bool
        For a single chain of molecules the average of M is zero. This flag sets
        <M> = 0.
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output_prefix files

    Attributes
    ----------
    ${CYLINDER_CLASS_ATTRIBUTES}
    results.eps_ax : numpy.ndarray
        Parallel dielectric profile (ε_∥)
    results.deps_ax : numpy.ndarray
        Error of parallel dielectric profile
    results.eps_rad : numpy.ndarray
        Inverse perpendicular dielectric profile (ε^{-1}_⟂)
    results.deps_rad : numpy.ndarray
        Error of inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroup,
                 bin_width=0.1,
                 temperature=300,
                 single=False,
                 output_prefix="eps_cyl",
                 refgroup=None,
                 concfreq=0,
                 dim=2,
                 rmin=0,
                 rmax=None,
                 zmin=None,
                 zmax=None,
                 vcutwidth=0.1,
                 unwrap=True):
        super(DielectricCylinder, self).__init__(atomgroup,
                                                 concfreq=concfreq,
                                                 refgroup=refgroup,
                                                 rmin=rmin,
                                                 rmax=rmax,
                                                 zmin=zmin,
                                                 zmax=zmax,
                                                 dim=dim,
                                                 bin_width=bin_width,
                                                 unwrap=unwrap)
        self.output_prefix = output_prefix
        self.temperature = temperature
        self.single = single
        self.vcutwidth = vcutwidth

    def _prepare(self):
        super(DielectricCylinder, self)._prepare()
        self.comp, ix = get_compound(self.atomgroup.atoms, return_index=True)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

    def _single_frame(self):
        super(DielectricCylinder, self)._single_frame()

        # Use polarization density (for radial component)
        # ========================================================
        rbins = np.digitize(self.pos_cyl[:, 0], self._obs.bin_edges[1:])

        curQ_rad, _ = np.histogram(rbins,
                                   bins=np.arange(self.n_bins + 1),
                                   weights=self.atomgroup.charges)

        self._obs.m_rad = -np.cumsum(
            (curQ_rad / self._obs.bin_volume) * self._obs.bin_pos
            * self._obs.bin_width) / self._obs.bin_pos
        self._obs.M_rad = np.sum(self._obs.m_rad * self._obs.bin_width)
        self._obs.mM_rad = self._obs.m_rad * self._obs.M_rad
        # Use virtual cutting method ( for axial component )
        # ========================================================
        # number of virtual cuts ("many")
        nbinsz = np.ceil(self._obs.L / self.vcutwidth).astype(int)

        # Move all r-positions to 'center of charge' such that we avoid
        # monopoles in r-direction. We only want to cut in z direction.
        chargepos = self.pos_cyl[:, 0] * np.abs(self.atomgroup.charges)
        center = (self.atomgroup.accumulate(chargepos, compound=self.comp)
                  / self.atomgroup.accumulate(np.abs(self.atomgroup.charges),
                                              compound=self.comp))
        testpos = center[self.inverse_ix]
        rbins = np.digitize(testpos, self._obs.bin_edges[1:])
        z = (np.arange(nbinsz) + 1) * (self._obs.L / nbinsz)
        zbins = np.digitize(self.pos_cyl[:, 2], z)

        curQz, _, _ = np.histogram2d(
            zbins, rbins,
            bins=[np.arange(nbinsz + 1), np.arange(self.n_bins + 1)],
            weights=self.atomgroup.charges)

        curqz = np.cumsum(curQz, axis=0) / (self._obs.bin_area)[np.newaxis, :]
        self._obs.m_ax = -curqz.mean(axis=0)
        self._obs.M_ax = np.dot(self.atomgroup.charges, self.pos_cyl[:, 2])
        self._obs.mM_ax = self._obs.m_ax * self._obs.M_ax

    def _conclude(self):
        super(DielectricCylinder, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        if not self.single:
            cov_ax = self.means.mM_ax - self.means.m_ax * self.means.M_ax
            cov_rad = self.means.mM_rad - self.means.m_rad * self.means.M_rad

            dcov_ax = 0.5 * np.sqrt(
                self.sems.mM_ax**2 + self.sems.m_ax**2 * self.means.M_ax**2
                + self.means.m_ax**2 * self.sems.M_ax**2)
            dcov_rad = 0.5 * np.sqrt(
                self.sems.mM_rad**2 + self.sems.m_rad**2 * self.means.M_rad**2
                + self.means.m_rad**2 * self.sems.M_rad**2)
        else:
            # <M> = 0 for a single line of water molecules.
            cov_ax = self.means.mM_ax
            cov_rad = self.means.mM_rad
            dcov_ax = self.sems.mM_ax
            dcov_rad = self.sems.mM_rad

        self.results.eps_ax = 1 + pref * cov_ax
        self.results.deps_ax = pref * dcov_ax

        self.results.eps_rad = 1 - (2 * np.pi * self._obs.L
                                    * pref * self.results.bin_pos * cov_rad)
        self.results.deps_rad = (2 * np.pi * self._obs.L
                                 * pref * self.results.bin_pos * dcov_rad)

    def save(self):
        """Save result."""
        outdata_ax = np.array([
            self.results.bin_pos, self.results.eps_ax, self.results.deps_ax
            ]).T
        outdata_rad = np.array([
            self.results.bin_pos, self.results.eps_rad, self.results.deps_rad
            ]).T

        columns = ["positions [Å]"]

        columns += ["eps_ax", "eps_ax error"]

        self.savetxt("{}{}".format(self.output_prefix, "_ax.dat"),
                     outdata_ax,
                     columns=columns)

        columns = ["positions [Å]"]

        columns += ["eps_rad", "eps_rad error"]

        self.savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                     outdata_rad,
                     columns=columns)


@render_docs
@charge_neutral(filter="error")
class DielectricSphere(SphereBase):
    """Calculate spherical dielectric profiles.

    Components are calculated along and radial direction at the
    system's center of mass.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${SPHERE_CLASS_PARAMETERS}
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output_prefix files

    Attributes
    ----------
    ${RADIAL_CLASS_ATTRIBUTES}
    results.eps_rad : numpy.ndarray
        Inverse perpendicular dielectric profile (ε^{-1}_⟂)
    results.deps_rad : numpy.ndarray
        Error of inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroup,
                 bin_width=0.1,
                 temperature=300,
                 output_prefix="eps_sph",
                 refgroup=None,
                 concfreq=0,
                 rmin=0,
                 rmax=None,
                 unwrap=True):
        super(DielectricSphere, self).__init__(atomgroup,
                                               concfreq=concfreq,
                                               refgroup=refgroup,
                                               rmin=rmin,
                                               rmax=rmax,
                                               bin_width=bin_width,
                                               unwrap=unwrap)
        self.output_prefix = output_prefix
        self.bin_width = bin_width
        self.temperature = temperature

    def _prepare(self):
        super(DielectricSphere, self)._prepare()
        self.comp, ix = get_compound(self.atomgroup.atoms, return_index=True)
        _, self.inverse_ix = np.unique(ix, return_inverse=True)

    def _single_frame(self):
        super(DielectricSphere, self)._single_frame()
        rbins = np.digitize(
            self.transform_positions(self.atomgroup.positions)[:, 0],
            self._obs.bin_edges[1:])

        curQ_rad, _ = np.histogram(rbins,
                                   bins=np.arange(self.n_bins + 1),
                                   weights=self.atomgroup.charges)

        self._obs.m_rad = -np.cumsum(
            (curQ_rad / self._obs.bin_volume) * self._obs.bin_pos**2
            * self._obs.bin_width) / self._obs.bin_pos**2
        self._obs.M_rad = np.sum(self._obs.m_rad * self._obs.bin_width)
        self._obs.mM_rad = self._obs.m_rad * self._obs.M_rad

    def _conclude(self):
        super(DielectricSphere, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        cov_rad = self.means.mM_rad - self.means.m_rad * self.means.M_rad

        dcov_rad = 0.5 * np.sqrt(
            self.sems.mM_rad**2 + self.sems.m_rad**2 * self.means.M_rad**2
            + self.means.m_rad**2 * self.sems.M_rad**2)

        self.results.eps_rad = 1 - (4 * np.pi * self.results.bin_pos**2
                                    * pref * cov_rad)
        self.results.deps_rad = (4 * np.pi * self.results.bin_pos**2
                                 * pref * dcov_rad)

    def save(self):
        """Save result."""
        outdata_rad = np.array([
            self.results.bin_pos, self.results.eps_rad, self.results.deps_rad
            ]).T

        columns = ["positions [Å]", "eps_rad", "eps_rad error"]

        self.savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                     outdata_rad,
                     columns=columns)


@render_docs
@charge_neutral(filter="error")
class DielectricSpectrum(AnalysisBase):
    r"""Compute the linear dielectric spectrum.

    This module, given molecular dynamics trajectory data, produces a
    `.txt` file containing the complex dielectric function as a function of
    the (linear, not radial -
    i.e. :math:`\nu` or :math:`f`, rather than :math:`\omega`) frequency, along
    with the associated standard deviations.
    The algorithm is based on the Fluctuation Dissipation Relation (FDR):
    :math:`\chi(f) = -1/(3 V k_B T \varepsilon_0)
    FT[\theta(t) \langle P(0) dP(t)/dt\rangle]`.
    By default, the polarization trajectory, time series array and the average
    system volume are saved in the working directory, and the data are
    reloaded from these files if they are present.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${BASE_CLASS_PARAMETERS}
    temperature : float
        Reference temperature.
    output_prefix : str
        Prefix for the output files.
    segs : int
        Sets the number of segments the trajectory is broken into.
    df : float
        The desired frequency spacing in THz.
        This determines the minimum frequency about which there
        is data. Overrides `segs` option.
    bins : int
        Determines the number of bins used for data averaging;
        (this parameter sets the upper limit).
        The data are by default binned logarithmically.
        This helps to reduce noise, particularly in
        the high-frequency domain, and also prevents plot
        files from being too large.
    binafter : int
        The number of low-frequency data points that are
        left unbinned.
    nobin : bool
        Prevents the data from being binned altogether. This
        can result in very large plot files and errors.

    Attributes
    ----------
    results
    """

    # TODO: set up script to calc spectrum at intervals while calculating
    # polarization for very big-data trajectories
    # TODO: merge with molecular version?
    def __init__(self,
                 atomgroup,
                 unwrap=True,
                 temperature=300,
                 output_prefix="",
                 segs=20,
                 df=None,
                 bins=200,
                 binafter=20,
                 nobin=False):
        super(DielectricSpectrum, self).__init__(atomgroup,
                                                 unwrap=unwrap)
        self.temperature = temperature
        self.output_prefix = output_prefix
        self.segs = segs
        self.df = df
        self.bins = bins
        self.binafter = binafter
        self.nobin = nobin

    def _prepare(self):
        if len(self.output_prefix) > 0:
            self.output_prefix += "_"

        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P = np.zeros((self.n_frames, 3))

    def _single_frame(self):
        self.V += self._ts.volume
        self.P[self._frame_index, :] = np.dot(self.atomgroup.charges,
                                              self.atomgroup.positions)

    def _conclude(self):
        self.results.t = self._trajectory.dt * self.frames
        self.results.V = self.V / self._index

        self.results.P = self.P

        # Find a suitable number of segments if it's not specified:
        if self.df is not None:
            self.segs = np.max([int(self.n_frames * self.dt * self.df), 2])

        self.seglen = int(self.n_frames / self.segs)

        # Prefactor for susceptibility:
        # Polarization: eÅ^2 to e m^2
        pref = (scipy.constants.e)**2 * scipy.constants.angstrom**2
        # Volume: Å^3 to m^3
        pref /= 3 * self.results.V * scipy.constants.angstrom**3
        pref /= scipy.constants.k * self.temperature
        pref /= scipy.constants.epsilon_0

        logger.info('Calculating susceptibilty and errors...')

        # if t too short to simply truncate
        if len(self.results.t) < 2 * self.seglen:
            self.results.t = np.append(
                self.results.t,
                self.results.t + self.results.t[-1] + self.dt)

        # truncate t array (it's automatically longer than 2 * seglen)
        self.results.t = self.results.t[:2 * self.seglen]
        # get freqs
        self.results.nu = FT(
            self.results.t,
            np.append(self.results.P[:self.seglen, 0],
                      np.zeros(self.seglen)))[0]
        # susceptibility
        self.results.susc = np.zeros(self.seglen, dtype=complex)
        # std deviation of susceptibility
        self.results.dsusc = np.zeros(self.seglen, dtype=complex)
        # susceptibility for current seg
        ss = np.zeros((2 * self.seglen), dtype=complex)

        # loop over segs
        for s in range(0, self.segs):
            logger.info(f'\rSegment {s + 1} of {self.segs}')
            ss = 0 + 0j

            # loop over x, y, z
            for self._i in range(3):
                FP = FT(
                    self.results.t,
                    np.append(
                        self.results.P[s * self.seglen:(s + 1)
                                         * self.seglen, self._i],
                        np.zeros(self.seglen)), False)
                ss += FP.real * FP.real + FP.imag * FP.imag

            ss *= self.results.nu * 1j

            # Get the real part by Kramers Kronig
            ss.real = iFT(
                self.results.t, 1j * np.sign(self.results.nu)
                                   * FT(self.results.nu, ss, False), False).imag

            if s == 0:
                self.results.susc += ss[self.seglen:]

            else:
                ds = ss[self.seglen:] - \
                    (self.results.susc / s)
                self.results.susc += ss[self.seglen:]
                dif = ss[self.seglen:] - \
                    (self.results.susc / (s + 1))
                ds.real *= dif.real
                ds.imag *= dif.imag
                # variance by Welford's Method
                self.results.dsusc += ds

        self.results.dsusc.real = np.sqrt(self.results.dsusc.real)
        self.results.dsusc.imag = np.sqrt(self.results.dsusc.imag)

        # 1/2 b/c it's the full FT, not only half-domain
        self.results.susc *= pref / (2 * self.seglen * self.segs * self.dt)
        self.results.dsusc *= pref / (2 * self.seglen * self.segs * self.dt)

        # Discard negative-frequency data; contains the same
        # information as positive regime:
        # Now nu represents positive f instead of omega
        self.results.nu = self.results.nu[self.seglen:] / (2 * np.pi)

        logger.info(f'Length of segments:    {self.seglen} frames,'
                    f' {self.seglen * self.dt:.0f} ps')
        logger.info(f'Frequency spacing:    '
                    f'~ {self.segs / (self.n_frames * self.dt):.5f} THz')

        # Bin data if there are too many points:
        if not (self.nobin or self.seglen <= self.bins):
            bins = np.logspace(
                np.log(self.binafter) / np.log(10),
                np.log(len(self.results.susc)) / np.log(10),
                self.bins - self.binafter + 1).astype(int)
            bins = np.unique(np.append(np.arange(self.binafter), bins))[:-1]

            self.results.nu_binned = bin(self.results.nu, bins)
            self.results.susc_binned = bin(self.results.susc, bins)
            self.results.dsusc_binned = bin(self.results.dsusc, bins)

            logger.info(f'Binning data above datapoint '
                        f'{self.binafter} in log-spaced bins')
            logger.info(f'Binned data consists of '
                        f'{len(self.results.susc)} datapoints')
        # data is binned
        logger.info(f'Not binning data: there are '
                    f'{len(self.results.susc)} datapoints')

    def save(self):
        """Save result."""
        np.save(self.output_prefix + 'tseries.npy', self.results.t)

        with open(self.output_prefix + 'V.txt', "w") as Vfile:
            Vfile.write(str(self.results.V))

        np.save(self.output_prefix + 'P_tseries.npy', self.results.P)

        suscfilename = "{}{}".format(self.output_prefix, 'susc.dat')
        self.savetxt(
            suscfilename,
            np.transpose([
                self.results.nu, self.results.susc.real,
                self.results.dsusc.real, self.results.susc.imag,
                self.results.dsusc.imag
                ]),
            columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)",
                                "Δ imag(χ)"])

        logger.info('Susceptibility data saved as ' + suscfilename)

        if not (self.nobin or self.seglen <= self.bins):

            suscfilename = "{}{}".format(self.output_prefix, 'susc_binned.dat')
            self.savetxt(suscfilename,
                         np.transpose([
                             self.results.nu_binned,
                             self.results.susc_binned.real,
                             self.results.dsusc_binned.real,
                             self.results.susc_binned.imag,
                             self.results.dsusc_binned.imag
                             ]),
                         columns=["ν [THz]", "real(χ)", " Δ real(χ)", "imag(χ)",
                                  "Δ imag(χ)"])

            logger.info('Binned susceptibility data saved as ' + suscfilename)
