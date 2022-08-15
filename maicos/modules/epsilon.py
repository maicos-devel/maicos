#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Tools for computing relative permittivity.

The dielectric constant modules of MAICoS allow for computing dielectric
profile and dielectric spectrum from molecular simulation trajectory files.
"""

import logging

import MDAnalysis as mda
import numpy as np
import scipy.constants

from ..decorators import (
    charge_neutral,
    make_whole,
    set_planar_class_doc,
    set_verbose_doc,
    )
from ..utils import FT, check_compound, iFT, savetxt, symmetrize
from .base import AnalysisBase, PlanarBase


logger = logging.getLogger(__name__)


def Bin(a, bins):
    """Average array values in bins for easier plotting.

    Note: "bins" array should contain the INDEX (integer)
    where that bin begins
    """
    if np.iscomplex(a).any():
        avg = np.zeros(len(bins), dtype=complex)  # average of data
    else:
        avg = np.zeros(len(bins))

    count = np.zeros(len(bins), dtype=int)
    ic = -1

    for i in range(0, len(a)):
        if i in bins:
            ic += 1  # index for new average
        avg[ic] += a[i]
        count[ic] += 1

    return avg / count


@set_verbose_doc
@set_planar_class_doc
@make_whole()
@charge_neutral(filter="error")
class EpsilonPlanar(PlanarBase):
    """Calculate planar dielectric profiles.

    See Schlaich, et al., Phys. Rev. Lett., vol. 117 (2016) for details.

    Parameters
    ----------
    atomgroups : list[AtomGroup]
        List of :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the dielectric profiles are calculated.
    ${PLANAR_CLASS_PARAMETERS}
    xy : bool
        Use 2D slab geometry.
    vac : bool
        Use vacuum boundary conditions instead of metallic (2D only!).
    sym : bool
        Symmetrize the profiles.
    ${MAKE_WHOLE_PARAMETER}
    temperature : float
        temperature (K)
    output_prefix : str
        Prefix for output files.
    vcutwidth : float
        Spacing of virtual cuts (bins) along the parallel directions.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.dens_mean : np.ndarray
        eps_par: Parallel dielectric profile (ε_∥ - 1)
    results.deps_par : np.ndarray
        Error of parallel dielectric profile
    results.eps_par_self : np.ndarray
        Self contribution of parallel dielectric profile
    results.eps_par_coll : np.ndarray
        Collective contribution of parallel dielectric profile
    results.eps_perp : np.ndarray
        Inverse perpendicular dielectric profile (ε^{-1}_⟂ - 1)
    results.deps_perp : np.ndarray
        Error of inverse perpendicular dielectric profile
    results.eps_par_self : np.ndarray
        Self contribution of Inverse perpendicular dielectric profile
    results.eps_perp_coll : np.ndarray
        Collective contribution of Inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroups,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=0.5,
                 center=False,
                 comgroup=None,
                 xy=False,
                 sym=False,
                 vac=False,
                 make_whole=True,
                 temperature=300,
                 output_prefix="eps",
                 concfreq=0,
                 vcutwidth=0.1,
                 **kwargs):
        super(EpsilonPlanar, self).__init__(atomgroups=atomgroups,
                                            dim=dim,
                                            zmin=zmin,
                                            zmax=zmax,
                                            binwidth=binwidth,
                                            center=center,
                                            comgroup=comgroup,
                                            multi_group=True,
                                            **kwargs)
        self.xy = xy
        self.sym = sym
        self.vac = vac
        self.make_whole = make_whole
        self.temperature = temperature
        self.output_prefix = output_prefix
        self.concfreq = concfreq
        self.vcutwidth = vcutwidth

    def _prepare(self):
        super(EpsilonPlanar, self)._prepare()
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]

        self.results.frame.V = 0

        self.results.frame.M_par = np.zeros(2)
        self.results.frame.M_perp = 0
        self.results.frame.M_perp_2 = 0

        n_ag = len(self.atomgroups)

        self.results.frame.m_par = np.zeros((self.n_bins, 2, n_ag))
        self.results.frame.mM_par = np.zeros((self.n_bins, n_ag))
        self.results.frame.mm_par = np.zeros((self.n_bins, n_ag))
        self.results.frame.cmM_par = np.zeros((self.n_bins, n_ag))
        self.results.frame.cM_par = np.zeros((self.n_bins, 2, n_ag))

        self.results.frame.m_perp = np.zeros((self.n_bins, n_ag))
        self.results.frame.mM_perp = np.zeros((self.n_bins, n_ag))
        self.results.frame.mm_perp = np.zeros((self.n_bins, n_ag))
        self.results.frame.cmM_perp = np.zeros((self.n_bins, n_ag))
        self.results.frame.cM_perp = np.zeros((self.n_bins, n_ag))

    def _single_frame(self):
        super(EpsilonPlanar, self)._single_frame()
        A = np.prod(self._universe.dimensions[self.xydims])
        dz = (self.zmax - self.zmin) / self.n_bins
        self.results.frame.V = (self.zmax - self.zmin) * A
        self.results.frame.V_bin = A * dz

        # precalculate total polarization of the box
        self.results.frame.M = np.dot(self._universe.atoms.charges,
                                      self._universe.atoms.positions)

        self.results.frame.M_perp = self.results.frame.M[self.dim]
        self.results.frame.M_perp_2 = self.results.frame.M[self.dim]**2
        self.results.frame.M_par = self.results.frame.M[self.xydims]

        # Use polarization density ( for perpendicular component )
        # ========================================================
        for i, sel in enumerate(self.atomgroups):
            zpos = np.zeros(len(sel))
            np.clip(sel.atoms.positions[:, self.dim],
                    self.zmin, self.zmax, zpos)

            curQ = np.histogram(zpos,
                                bins=self.n_bins,
                                range=[self.zmin, self.zmax],
                                weights=sel.atoms.charges)[0]

            self.results.frame.m_perp[:, i] = -np.cumsum(curQ / A)
            self.results.frame.mM_perp[:, i] = \
                self.results.frame.m_perp[:, i] * self.results.frame.M_perp
            self.results.frame.mm_perp[:, i] = \
                self.results.frame.m_perp[:, i]**2 * self.results.frame.V_bin
            self.results.frame.cmM_perp[:, i] = \
                self.results.frame.m_perp[:, i] \
                * (self.results.frame.M_perp
                   - self.results.frame.m_perp[:, i]
                   * self.results.frame.V_bin)

            self.results.frame.cM_perp[:, i] = \
                self.results.frame.M_perp \
                - self.results.frame.m_perp[:, i] \
                * self.results.frame.V_bin

            # Use virtual cutting method ( for parallel component )
            # ========================================================
            # Move all z-positions to 'center of charge' such
            # that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only
            # want to cut in x/y direction)
            comp = check_compound(sel.atoms)
            if comp == "molecules":
                repeats = np.unique(sel.atoms.molnums,
                                    return_counts=True)[1]
            elif comp == "fragments":
                repeats = np.unique(sel.atoms.fragindices,
                                    return_counts=True)[1]
            else:
                repeats = np.unique(sel.atoms.resids,
                                    return_counts=True)[1]

            if np.all(repeats == repeats[0]):
                # selection contains identical components and we can
                # supercharge the calculation. (mda.atomgroup.center
                # with the compound option is slow!)
                chargepos = (
                    np.abs(sel.charges)
                    * sel.atoms.positions[:, 2]
                    ).reshape(len(repeats), repeats[0])

                centers = np.sum(chargepos, axis=1) \
                    / np.sum(np.abs(sel.charges)[:repeats[0]])
            else:
                centers = sel.center(weights=np.abs(sel.charges),
                                     compound=check_compound(sel))[:, self.dim]

            testpos = np.repeat(centers, repeats)
            # Average parallel directions
            for j, direction in enumerate(self.xydims):
                # At this point we should not use the wrap, which causes
                # unphysical dipoles at the borders
                Lx = self._ts.dimensions[direction]
                Ax = self._ts.dimensions[self.xydims[1 - j]] * dz
                vbinsx = np.ceil(Lx / self.vcutwidth).astype(int)

                xpos = np.zeros(len(sel))
                np.clip(sel.atoms.positions[:, direction], 0, Lx, xpos)

                curQx = np.histogram2d(
                    testpos, xpos,
                    bins=[self.n_bins, vbinsx],
                    range=[[self.zmin, self.zmax], [0, Lx]],
                    weights=sel.atoms.charges)[0]

                # integral over x, so uniself._ts of area
                curqx = -np.cumsum(curQx / Ax, axis=1).mean(axis=1)
                self.results.frame.m_par[:, j, i] = curqx
            self.results.frame.mM_par[:, i] = \
                np.dot(self.results.frame.m_par[:, :, i],
                       self.results.frame.M_par)
            self.results.frame.mm_par[:, i] = (
                self.results.frame.m_par[:, :, i]
                * self.results.frame.m_par[:, :, i]
                ).sum(axis=1) \
                * self.results.frame.V_bin
            self.results.frame.cmM_par[:, i] = \
                (self.results.frame.m_par[:, :, i]
                 * (self.results.frame.M_par
                    - self.results.frame.m_par[:, :, i]
                    * self.results.frame.V_bin)
                 ).sum(axis=1)
            self.results.frame.cM_par[:, :, i] = \
                self.results.frame.M_par \
                - self.results.frame.m_par[:, :, i] \
                * self.results.frame.V_bin

        return self.results.frame.M_par[0]

    def _conclude(self):
        super(EpsilonPlanar, self)._conclude()

        pref = 1 / scipy.constants.epsilon_0
        pref /= scipy.constants.Boltzmann * self.temperature
        # Convert from ~e^2/m to ~base units
        pref /= scipy.constants.angstrom / \
            (scipy.constants.elementary_charge)**2

        self.results.pref = pref
        self.results.V = self.results.means.V

        cov_perp = self.results.means.mM_perp \
            - self.results.means.m_perp \
            * self.results.means.M_perp

        # Using propagation of uncertainties
        dcov_perp = np.sqrt(
            self.results.sems.mM_perp**2
            + (self.results.means.M_perp * self.results.sems.m_perp)**2
            + (self.results.means.m_perp * self.results.sems.M_perp)**2
            )

        var_perp = self.results.means.M_perp_2 - self.results.means.M_perp**2

        cov_perp_self = self.results.means.mm_perp \
            - (self.results.means.m_perp**2
               * self.results.means.V / self.n_bins)
        cov_perp_coll = self.results.means.cmM_perp \
            - self.results.means.m_perp * self.results.means.cM_perp

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
                - cov_perp / (pref**-1 + var_perp / self.results.means.V)
            self.results.deps_perp = pref * dcov_perp

            self.results.eps_perp_self = \
                (- pref * cov_perp_self) \
                / (1 + pref / self.results.means.V * var_perp)
            self.results.eps_perp_coll = \
                (- pref * cov_perp_coll) \
                / (1 + pref / self.results.means.V * var_perp)

        cov_par = np.zeros((self.n_bins, len(self.atomgroups)))
        dcov_par = np.zeros((self.n_bins, len(self.atomgroups)))
        cov_par_self = np.zeros((self.n_bins, len(self.atomgroups)))
        cov_par_coll = np.zeros((self.n_bins, len(self.atomgroups)))

        for i, _ in enumerate(self.atomgroups):
            cov_par[:, i] = 0.5 * (self.results.means.mM_par[:, i]
                                   - np.dot(self.results.means.m_par[:, :, i],
                                            self.results.means.M_par))

            # Using propagation of uncertainties
            dcov_par[:, i] = 0.5 * np.sqrt(
                self.results.sems.mM_par[:, i]**2
                + np.dot(self.results.sems.m_par[:, :, i]**2,
                         self.results.means.M_par**2)
                + np.dot(self.results.means.m_par[:, :, i]**2,
                         self.results.sems.M_par**2)
                )

            cov_par_self[:, i] = 0.5 * (
                self.results.means.mm_par[:, i]
                - np.dot(self.results.means.m_par[:, :, i],
                         self.results.means.m_par[:, :, i].sum(axis=0)))
            cov_par_coll[:, i] = \
                0.5 * (self.results.means.cmM_par[:, i]
                       - (self.results.means.m_par[:, :, i]
                       * self.results.means.cM_par[:, :, i]).sum(axis=1))

        self.results.eps_par = pref * cov_par
        self.results.deps_par = pref * dcov_par
        self.results.eps_par_self = pref * cov_par_self
        self.results.eps_par_coll = pref * cov_par_coll

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
            self.results.z[:, np.newaxis],
            self.results.eps_perp.sum(axis=1)[:, np.newaxis],
            self.results.eps_perp,
            np.linalg.norm(self.results.deps_perp,
                           axis=1)[:, np.newaxis], self.results.deps_perp,
            self.results.eps_perp_self.sum(axis=1)[:, np.newaxis],
            self.results.eps_perp_coll.sum(axis=1)[:, np.newaxis],
            self.results.eps_perp_self, self.results.eps_perp_coll
            ])
        outdata_par = np.hstack([
            self.results.z[:, np.newaxis],
            self.results.eps_par.sum(axis=1)[:, np.newaxis],
            self.results.eps_par,
            np.linalg.norm(self.results.deps_par,
                           axis=1)[:, np.newaxis], self.results.deps_par,
            self.results.eps_par_self.sum(axis=1)[:, np.newaxis],
            self.results.eps_par_coll.sum(axis=1)[:, np.newaxis],
            self.results.eps_par_self, self.results.eps_par_coll
            ])

        header = "statistics over {:.1f} picoseconds".format(
            self._index * self._universe.trajectory.dt)
        savetxt("{}{}".format(self.output_prefix, "_perp"),
                outdata_perp,
                header=header)
        savetxt("{}{}".format(self.output_prefix, "_par"),
                outdata_par,
                header=header)


@set_verbose_doc
@make_whole()
@charge_neutral(filter="error")
class EpsilonCylinder(AnalysisBase):
    """Calculate cylindrical dielectric profiles.

    Components are calculated along the axial (z) and radial (along xy)
    direction at the system's center of mass.

    Parameters
    ----------
    atomgroup : AtomGroup
        :class:`~MDAnalysis.core.groups.AtomGroup` for which
        the dielectric profiles are calculated
    geometry : str
        A structure file without water from which com is calculated.
    radius : float
        Radius of the cylinder (Å)
    binwidth : float
        Bindiwdth the binwidth (Å)
    variable_dr : bool
        Use a variable binwidth, where the volume is kept fixed.
    length : float
        Length of the cylinder (Å)
    ${MAKE_WHOLE_PARAMETER}
    temperature : float
        temperature (K)
    single : bool
        "1D" line of watermolecules
    output_prefix : str
        Prefix for output_prefix files
    concfreq : int
        Default number of frames after which results are calculated and
        files refreshed. If `0` results are only calculated at the end of
        the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.r : np.ndarray
        bins
    results.eps_ax : np.ndarray
        Parallel dielectric profile (ε_∥)
    results.deps_ax : np.ndarray
        Error of parallel dielectric profile
    results.eps_rad : np.ndarray
        Inverse perpendicular dielectric profile (ε^{-1}_⟂)
    results.deps_rad : np.ndarray
        Error of inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroups,
                 binwidth=0.5,
                 geometry=None,
                 radius=None,
                 variable_dr=False,
                 length=None,
                 temperature=300,
                 single=False,
                 make_whole=True,
                 output_prefix="eps_cyl",
                 concfreq=0,
                 **kwargs):
        super(EpsilonCylinder, self).__init__(atomgroups, **kwargs)
        self.output_prefix = output_prefix
        self.binwidth = binwidth
        self.geometry = geometry
        self.radius = radius
        self.variable_dr = variable_dr
        self.length = length
        self.temperature = temperature
        self.single = single
        self.make_whole = make_whole
        self.concfreq = concfreq

    def _prepare(self):
        if self.geometry is not None:
            self.com = self.system.atoms.center_of_mass(
                mda.Universe(self.geometry))
        else:
            logger.info("No geometry set. "
                        "Calculate center of geometry from box dimensions.")
            self.com = self._universe.dimensions[:3] / 2

        if self.radius is None:
            logger.info("No radius set. Take smallest box extension.")
            self.radius = self._universe.dimensions[:2].min() / 2

        if self.length is None:
            self.length = self._universe.dimensions[2]

        self.n_bins = int(np.ceil(self.radius / self.binwidth))

        if self.variable_dr:
            # variable dr
            sol = np.ones(self.n_bins) * self.radius**2 / self.n_bins
            mat = np.diag(np.ones(self.n_bins)) + np.diag(
                np.ones(self.n_bins - 1) * -1, k=-1)

            self.r_bins = np.sqrt(np.linalg.solve(mat, sol))
            self.dr = self.r_bins - np.insert(self.r_bins, 0, 0)[0:-1]
        else:
            # Constant dr
            self.dr = np.ones(self.n_bins) * self.radius / self.n_bins
            self.r_bins = np.arange(self.n_bins) * self.dr + self.dr

        self.delta_r_sq = self.r_bins**2 - np.insert(self.r_bins, 0,
                                                     0)[0:-1]**2  # r_o^2-r_i^2
        self.r = np.copy(self.r_bins) - self.dr / 2

        self.results.r = self.r

        # Use resampling for error estimation.
        # We do block averaging for 10 hardcoded blocks.
        self.resample = 10
        self.resample_freq = int(np.ceil(self.n_frames / self.resample))

        self.m_rad = np.zeros((self.n_bins, self.resample))

        self.M_rad = np.zeros((self.resample))
        self.mM_rad = np.zeros(
            (self.n_bins, self.resample))  # total fluctuations

        self.m_ax = np.zeros((self.n_bins, self.resample))
        self.M_ax = np.zeros((self.resample))
        self.mM_ax = np.zeros((self.n_bins, self.resample)
                              )  # total fluctuations

        logger.info('Using', self.n_bins, 'bins.')

    def _single_frame(self):
        # Transform from cartesian coordinates [x,y,z] to cylindrical
        # coordinates [r,z] (skip phi because of symmetry)
        positions_cyl = np.empty([self.atomgroup.positions.shape[0], 2])
        positions_cyl[:, 0] = np.linalg.norm(
            (self.atomgroup.positions[:, 0:2] - self.com[0:2]), axis=1)
        positions_cyl[:, 1] = self.atomgroup.positions[:, 2]
        positions_cyl[:, 1] %= self._ts.dimensions[2]

        # Use polarization density ( for radial component )
        # ========================================================
        curQ_rad = np.histogram(positions_cyl[:, 0],
                                bins=self.n_bins,
                                range=(0, self.radius),
                                weights=self.atomgroup.charges)[0]
        this_m_rad = -np.cumsum(
            (curQ_rad / self.delta_r_sq) * self.r * self.dr) / (self.r * np.pi
                                                                * self.length)

        this_M_rad = np.sum(this_m_rad * self.dr)
        self.M_rad[self._frame_index // self.resample_freq] += this_M_rad

        self.m_rad[:, self._frame_index // self.resample_freq] += this_m_rad
        self.mM_rad[:, self._frame_index
                    // self.resample_freq] += this_m_rad * this_M_rad

        # Use virtual cutting method ( for axial component )
        # ========================================================
        nbinsz = 250  # number of virtual cuts ("many")

        this_M_ax = np.dot(self.atomgroup.charges, positions_cyl[:, 1])
        self.M_ax[self._frame_index // self.resample_freq] += this_M_ax

        # Move all r-positions to 'center of charge' such that we avoid
        # monopoles in r-direction. We only want to cut in z direction.
        chargepos = positions_cyl * np.abs(
            self.atomgroup.charges[:, np.newaxis])
        centers = self.atomgroup.accumulate(
            chargepos, compound=check_compound(self.atomgroup))
        centers /= self.atomgroup.accumulate(
            np.abs(self.atomgroup.charges),
            compound=check_compound(self.atomgroup))[:, np.newaxis]
        comp = check_compound(self.atomgroup)
        if comp == "molecules":
            repeats = np.unique(self.atomgroup.molnums, return_counts=True)[1]
        elif comp == "fragments":
            repeats = np.unique(self.atomgroup.fragindices,
                                return_counts=True)[1]
        else:
            repeats = np.unique(self.atoms.resids, return_counts=True)[1]
        testpos = np.empty(positions_cyl[:, 0].shape)
        testpos = np.repeat(centers[:, 0], repeats)

        curQz = np.histogram2d(
            testpos,
            positions_cyl[:, 1],
            bins=(self.n_bins, nbinsz),
            range=((0, self.radius),
                   (0, self._ts.dimensions[2])),
            weights=self.atomgroup.charges)[0]
        curqz = np.cumsum(curQz,
                          axis=1) / (np.pi * self.delta_r_sq)[:, np.newaxis]

        this_m_ax = -curqz.mean(axis=1)

        self.m_ax[:, self._frame_index // self.resample_freq] += this_m_ax
        self.mM_ax[:, self._frame_index
                   // self.resample_freq] += this_m_ax * this_M_ax

    def _conclude(self):
        eps0inv = 1. / scipy.constants.epsilon_0
        pref = (scipy.constants.elementary_charge)**2 / 1e-10

        if self.single:  # removed average of M if single line water.
            cov_ax = self.mM_ax.sum(axis=1) / self._index
            cov_rad = self.mM_rad.sum(axis=1) / self._index

            dcov_ax = (self.mM_ax.std(axis=1) / self._index * self.resample) \
                / np.sqrt(self.resample - 1)
            dcov_rad = (self.mM_rad.std(axis=1) / self._index * self.resample) \
                / np.sqrt(self.resample - 1)
        else:
            cov_ax = self.mM_ax.sum(axis=1) / self._index - \
                self.m_ax.sum(axis=1) / self._index * \
                self.M_ax.sum() / self._index
            cov_rad = self.mM_rad.sum(axis=1) / self._index - \
                self.m_rad.sum(axis=1) / self._index * \
                self.M_rad.sum() / self._index

            dcov_ax = np.sqrt(
                (self.mM_ax.std(axis=1) / self._index * self.resample)**2
                + (self.m_ax.std(axis=1) / self._index * self.resample
                   * self.M_ax.sum() / self._index)**2
                + (self.m_ax.sum(axis=1) / self._index * self.M_ax.std()
                   / self._index * self.resample)**2
                ) / np.sqrt(self.resample - 1)
            dcov_rad = np.sqrt((self.mM_rad.std(axis=1)
                                / self._index * self.resample)**2
                               + (self.m_rad.std(axis=1)
                                  / self._index * self.resample
                                  * self.M_rad.sum() / self._index)**2
                               + (self.m_rad.sum(axis=1)
                                  / self._index * self.M_rad.std()
                                  / self._index * self.resample)**2
                               ) / np.sqrt(self.resample - 1)

        beta = 1 / (scipy.constants.Boltzmann * self.temperature)

        self.results.eps_ax = 1 + beta * eps0inv * pref * cov_ax
        self.results.deps_ax = beta * eps0inv * pref * dcov_ax

        self.results.eps_rad = 1 - beta * eps0inv * pref * \
            2 * np.pi * self.r * self.length * cov_rad
        self.results.deps_rad = beta * eps0inv * pref * \
            2 * np.pi * self.r * self.length * dcov_rad

    def save(self):
        """Save result."""
        outdata_ax = np.array([
            self.results.r, self.results.eps_ax, self.results.deps_ax
            ]).T
        outdata_rad = np.array([
            self.results.r, self.results.eps_rad, self.results.deps_rad
            ]).T

        header = "statistics over {:.1f} picoseconds".format(
            self._index * self._universe.trajectory.dt)
        savetxt("{}{}".format(self.output_prefix, "_ax.dat"),
                outdata_ax,
                header=header)
        savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                outdata_rad,
                header=header)


@set_verbose_doc
@make_whole()
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
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    ${MAKE_WHOLE_PARAMETER}
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
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results
    """

    # TODO: set up script to calc spectrum at intervals while calculating
    # polarization for very big-data trajectories
    # TODO: merge with molecular version?
    def __init__(self,
                 atomgroup,
                 make_whole=True,
                 temperature=300,
                 output_prefix="",
                 segs=20,
                 df=None,
                 bins=200,
                 binafter=20,
                 nobin=False,
                 **kwargs):
        super(DielectricSpectrum, self).__init__(atomgroup, **kwargs)
        self.temperature = temperature
        self.make_whole = make_whole
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

            self.results.nu_binned = Bin(self.results.nu, bins)
            self.results.susc_binned = Bin(self.results.susc, bins)
            self.results.dsusc_binned = Bin(self.results.dsusc, bins)

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

        suscfilename = "{}{}".format(self.output_prefix, 'susc')
        savetxt(
            suscfilename,
            np.transpose([
                self.results.nu, self.results.susc.real,
                self.results.dsusc.real, self.results.susc.imag,
                self.results.dsusc.imag
                ]),
            delimiter='\t',
            header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

        logger.info('Susceptibility data saved as ' + suscfilename)

        if not (self.nobin or self.seglen <= self.bins):

            suscfilename = "{}{}".format(self.output_prefix, 'susc_binned')
            savetxt(suscfilename,
                    np.transpose([
                        self.results.nu_binned,
                        self.results.susc_binned.real,
                        self.results.dsusc_binned.real,
                        self.results.susc_binned.imag,
                        self.results.dsusc_binned.imag
                        ]),
                    delimiter='\t',
                    header="freq\tsusc\'\tstd_dev_susc\'\t-"
                    "susc\'\'\tstd_dev_susc\'\'")

            logger.info('Binned susceptibility data saved as ' + suscfilename)
