#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
import scipy.constants
import MDAnalysis as mda

from .base import SingleGroupAnalysisBase, MultiGroupAnalysisBase
from ..utils import check_compound, FT, iFT, savetxt
from ..decorators import charge_neutral, planar_base

eps0inv = 1. / scipy.constants.epsilon_0
pref = (scipy.constants.elementary_charge)**2 / 1e-10


def Bin(a, bins):
    """Averages array values in bins for easier plotting.
    Note: "bins" array should contain the INDEX (integer) where that bin begins"""

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


@charge_neutral(filter="default")
class epsilon_bulk(SingleGroupAnalysisBase):
    r"""Compute dipole moment fluctuations and static dielectric constant.

    **Inputs**

    :param outfreq (float): Number of frames after which the output is updated.
    :param output (str): Output filename.
    :param temperature (float): temperature (K)
    :param bpbc (bool): do not make broken molecules whole again
                               (only works if molecule is smaller than shortest
                                box vector

    **Outputs**

   :returns (dict): * M: Directional dependant dipole moment
                        :math:`\langle \boldsymbol M \rangle` in :math:`eÅ`.
                    * M2: Directional dependant squared dipole moment
                        :math:`\langle \boldsymbol M^2 \rangle` in :math:`(eÅ)^2`
                    * fluct: Directional dependant dipole moment fluctuation
                            :math:`\langle \boldsymbol M^2 \rangle - \langle \boldsymbol M \rangle^2`
                            in :math:`(eÅ)^2`
                    * eps: Directional dependant static dielectric constant
                    * eps_mean: Static dielectric constant
    """

    def __init__(self,
                 atomgroup,
                 outfreq=100,
                 temperature=300,
                 bpbc=True,
                 output="eps.dat",
                 **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.outfreq = 100
        self.temperature = temperature
        self.bpbc = bpbc
        self.output = output

    def _configure_parser(self, parser):
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-o', dest='output')
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument('-nopbcrepair', dest='bpbc')

    def _prepare(self):
        self.volume = 0
        self.M = np.zeros(3)
        self.M2 = np.zeros(3)
        self.charges = self.atomgroup.charges

    def _single_frame(self):
        # Make molecules whole
        if self.bpbc:
            self.atomgroup.unwrap(compound=check_compound(self.atomgroup))

        self.volume += self._ts.volume

        M = np.dot(self.charges, self.atomgroup.positions)
        self.M += M
        self.M2 += M * M

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        index = self._frame_index + 1
        beta = 1. / (scipy.constants.Boltzmann * self.temperature)

        self.results["M"] = self.M / index
        self.results["M2"] = self.M2 / index
        self.results["volume"] = self.volume / index
        self.results["fluct"] = self.results["M2"] - self.results["M"]**2
        self.results["eps"] = beta * eps0inv * pref * self.results["fluct"] / \
                              self.results["volume"]
        self.results["eps_mean"] = self.results["eps"].mean()

        self.results["eps"] += 1
        self.results["eps_mean"] += 1

    def _conclude(self):
        if self._verbose:
            print(
                "The following averages for the complete trajectory have been calculated:"
            )

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}> = {:.4f} eÅ".format(d, self.results["M"][i]))

            print("")
            for i, d in enumerate("xyz"):
                print(" <M_{}²> = {:.4f} (eÅ)²".format(d,
                                                       self.results["M2"][i]))

            print("")
            print(" <|M|²> = {:.4f} (eÅ)²".format(self.results["M2"].mean()))
            print(" |<M>|² = {:.4f} (eÅ)²".format(
                (self.results["M"]**2).mean()))

            print("")
            print(" <|M|²> - |<M>|² = {:.4f} (eÅ)²".format(
                self.results["fluct"].mean()))

            print("")
            for i, d in enumerate("xyz"):
                print(" ε_{} = {:.2f} ".format(d, self.results["eps"][i]))

            print("")
            print(" ε = {:.2f}".format(self.results["eps_mean"]))
            print("")

    def _save_results(self):
        savetxt(self.output,
                np.hstack([self.results["eps_mean"], self.results["eps"]]).T,
                fmt='%1.2f',
                header='eps\teps_x\teps_y\teps_z')


@charge_neutral(filter="error")
@planar_base()
class epsilon_planar(MultiGroupAnalysisBase):
    """Calculate planar dielectric profiles.

    See Schlaich, et al., Phys. Rev. Lett., vol. 117 (2016) for details

    **Inputs**

    :param output_prefix (str): Prefix for output files
    :param zmin (float): minimal coordinate for evaluation (nm)
    :param zmax (float): maximal coordinate for evaluation (nm)
    :param temperature (float): temperature (K)
    :param outfreq (int): Default number of frames after which output files are refreshed.
    :param b2d (bool): Use 2d slab geometry
    :param vac (bool): Use vacuum boundary conditions instead of metallic (2D only!).
    :param bsym (bool): symmetrize the profiles
    :param bpbc (bool): Do not make broken molecules whole again (only works if
                        molecule is smaller than shortest box vector

    **Outputs**

    :returns (dict): * z: Bin positions
                     * eps_par: Parallel dielectric profile (ε_∥ - 1)
                     * deps_par: Error of parallel dielectric profile
                     * eps_par_self: Self contribution of parallel dielectric profile
                     * eps_par_coll: Collective contribution of parallel dielectric profile
                     * eps_perp: Inverse perpendicular dielectric profile (ε^{-1}_⟂ - 1)
                     * deps_perp: Error of inverse perpendicular dielectric profile
                     * eps_par_self: Self contribution of Inverse perpendicular dielectric profile
                     * eps_perp_coll: Collective contribution of Inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroups,
                 output_prefix="eps",
                 zmin=0,
                 zmax=-1,
                 temperature=300,
                 outfreq=10000,
                 b2d=False,
                 bsym=False,
                 vac=False,
                 bpbc=True,
                 # Planar base arguments are necessary for buidling CLI
                 dim=2,
                 binwidth=0.05,
                 center=False,
                 comgroup=None,
                 **kwself):
        super().__init__(atomgroups, **kwself)
        self.output_prefix = output_prefix
        self.zmin = zmin
        self.zmax = zmax
        self.temperature = temperature
        self.outfreq = outfreq
        self.b2d = b2d
        self.bsym = bsym
        self.vac = vac
        self.bpbc = bpbc

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output_prefix')
        parser.add_argument('-zmin', dest='zmin')
        parser.add_argument('-zmax', dest='zmax')
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-2d', dest='b2d')
        parser.add_argument('-vac', dest='vac')
        parser.add_argument('-sym', dest='bsym')
        parser.add_argument('-nopbcrepair', dest='bpbc')

    def _prepare(self):
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]

        # Use 10 hardoced blocks for resampling
        self.resample = 10
        self.resample_freq = int(
            np.ceil((self.stopframe - self.startframe) / self.resample))

        self.V = 0
        self.A = np.prod(self._universe.dimensions[self.xydims])

        self.m_par = np.zeros((self.n_bins, len(self.atomgroups), self.resample))
        self.mM_par = np.zeros((self.n_bins, len(self.atomgroups),
                                self.resample))  # total fluctuations
        self.mm_par = np.zeros((self.n_bins, len(self.atomgroups)))  # self
        self.cmM_par = np.zeros(
            (self.n_bins, len(self.atomgroups)))  # collective contribution
        self.cM_par = np.zeros((self.n_bins, len(self.atomgroups)))
        self.M_par = np.zeros((self.resample))

        # Same for perpendicular
        self.m_perp = np.zeros(
            (self.n_bins, len(self.atomgroups), self.resample))
        self.mM_perp = np.zeros((self.n_bins, len(self.atomgroups),
                                 self.resample))  # total fluctuations
        self.mm_perp = np.zeros((self.n_bins, len(self.atomgroups)))  # self
        self.cmM_perp = np.zeros(
            (self.n_bins, len(self.atomgroups)))  # collective contribution
        self.cM_perp = np.zeros(
            (self.n_bins, len(self.atomgroups)))  # collective contribution
        self.M_perp = np.zeros((self.resample))
        self.M_perp_2 = np.zeros((self.resample))

    def _single_frame(self):

        if self.zmax == -1:
            zmax = self._ts.dimensions[self.dim]
        else:
            zmax = self.zmax

        if self.bpbc:
            # make broken molecules whole again!
            self._universe.atoms.unwrap(compound=check_compound(self._universe.atoms))

        dz_frame = self._ts.dimensions[self.dim] / self.n_bins

        # precalculate total polarization of the box
        this_M_perp, this_M_par = np.split(
            np.roll(
                np.dot(self._universe.atoms.charges,
                       self._universe.atoms.positions), -self.dim), [1])

        # Use polarization density ( for perpendicular component )
        # ========================================================

        # sum up the averages
        self.M_perp[self._frame_index // self.resample_freq] += this_M_perp
        self.M_perp_2[self._frame_index // self.resample_freq] += this_M_perp**2
        for i, sel in enumerate(self.atomgroups):
            bins = self.get_bins(sel.atoms.positions)
            curQ = np.histogram(bins,
                                bins=np.arange(self.n_bins + 1),
                                weights=sel.atoms.charges)[0]
            this_m_perp = -np.cumsum(curQ / self.A)
            self.m_perp[:, i, self._frame_index //
                        self.resample_freq] += this_m_perp
            self.mM_perp[:, i, self._frame_index //
                         self.resample_freq] += this_m_perp * this_M_perp
            self.mm_perp[:, i] += this_m_perp * this_m_perp * \
                (self._ts.dimensions[self.dim] / self.n_bins) * self.A  # self term
            # collective contribution
            self.cmM_perp[:, i] += this_m_perp * \
                (this_M_perp - this_m_perp * (self.A * dz_frame))
            self.cM_perp[:, i] += this_M_perp - this_m_perp * self.A * dz_frame

        # Use virtual cutting method ( for parallel component )
        # ========================================================
        nbinsx = 250  # number of virtual cuts ("many")

        for i, sel in enumerate(self.atomgroups):
            # Move all z-positions to 'center of charge' such that we avoid monopoles in z-direction
            # (compare Eq. 33 in Bonthuis 2012; we only want to cut in x/y direction)
            centers = sel.center(weights=np.abs(sel.charges),
                                 compound=check_compound(sel))
            comp = check_compound(sel.atoms)
            if comp == "molecules":
                repeats = np.unique(sel.atoms.molnums, return_counts=True)[1]
            elif comp == "fragments":
                repeats = np.unique(sel.atoms.fragindices, return_counts=True)[1]
            else:
                repeats = np.unique(sel.atoms.resids, return_counts=True)[1]
            testpos = sel.atoms.positions
            testpos[:, self.dim] = np.repeat(centers[:, self.dim], repeats)
            binsz = self.get_bins(testpos)

            # Average parallel directions
            for j, direction in enumerate(self.xydims):
                binsx = self.get_bins(sel.atoms.positions, dim=direction)
                curQx = np.histogram2d(binsz,
                                       binsx,
                                       bins=[
                                           np.arange(0, self.n_bins + 1),
                                           np.arange(0, nbinsx + 1)
                                       ],
                                       weights=sel.atoms.charges)[0]
                curqx = np.cumsum(curQx, axis=1) / (
                    self._ts.dimensions[self.xydims[1 - j]] *
                    (self._ts.dimensions[self.dim] / self.n_bins)
                )  # integral over x, so uniself._ts of area
                this_m_par = -curqx.mean(axis=1)

                self.m_par[:, i, self._frame_index //
                           self.resample_freq] += this_m_par
                self.mM_par[:, i, self._frame_index // self.resample_freq] += this_m_par * \
                    this_M_par[j]
                self.M_par[self._frame_index //
                           self.resample_freq] += this_M_par[j]
                self.mm_par[:, i] += this_m_par * \
                    this_m_par * dz_frame * self.A
                # collective contribution
                self.cmM_par[:, i] += this_m_par * \
                    (this_M_par[j] - this_m_par * dz_frame * self.A)
                self.cM_par[:, i] += this_M_par[j] - \
                    this_m_par * dz_frame * self.A

        self.V += self._ts.volume

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["V"] = self.V / self._index

        cov_perp = self.mM_perp.sum(axis=2) / self._index - \
            self.m_perp.sum(axis=2) / self._index * \
            self.M_perp.sum() / self._index
        dcov_perp = np.sqrt(
            (self.mM_perp.std(axis=2) / self._index * self.resample)**2 +
            (self.m_perp.std(axis=2) / self._index * self.resample *
             self.M_perp.sum() / self._index)**2 +
            (self.m_perp.sum(axis=2) / self._index * self.M_perp.std() /
             self._index * self.resample)**2) / np.sqrt(self.resample - 1)
        cov_perp_self = self.mm_perp / self._index - \
            (self.m_perp.sum(axis=2) / self._index * self.m_perp.sum(axis=2)
             / self._index * self.A * self.Lz / self.n_bins / self._index)
        cov_perp_coll = self.cmM_perp / self._index - \
            self.m_perp.sum(axis=2) / self._index * self.cM_perp / self._index

        var_perp = self.M_perp_2.sum() / self._index - (self.M_perp.sum() /
                                                        self._index)**2
        dvar_perp = (self.M_perp_2 / self._index - (self.M_perp / self._index)**2).std() \
            / np.sqrt(self.resample - 1)

        cov_par = self.mM_par.sum(axis=2) / self._index - \
            self.m_par.sum(axis=2) / self._index * \
            self.M_par.sum() / self._index
        cov_par_self = self.mm_par / self._index - \
            self.m_par.sum(axis=2) / self._index * (self.m_par.sum(axis=2) *
                                                    self.Lz / self.n_bins / self._index * self.A) / self._index
        cov_par_coll = self.cmM_par / self._index - \
            self.m_par.sum(axis=2) / self._index * self.cM_par / self._index
        dcov_par = np.sqrt(
            (self.mM_par.std(axis=2) / self._index * self.resample)**2 +
            (self.m_par.std(axis=2) / self._index * self.resample *
             self.M_par.sum() / self._index)**2 +
            (self.m_par.sum(axis=2) / self._index * self.M_par.std() /
             self._index * self.resample)**2) / np.sqrt(self.resample - 1)

        beta = 1 / (scipy.constants.Boltzmann * self.temperature)

        self.results["eps_par"] = beta * eps0inv * pref / 2 * cov_par
        self.results["deps_par"] = beta * eps0inv * pref / 2 * dcov_par
        self.results["eps_par_self"] = beta * eps0inv * pref / 2 * cov_par_self
        self.results["eps_par_coll"] = beta * eps0inv * pref / 2 * cov_par_coll

        if (self.b2d):
            self.results["eps_perp"] = -beta * eps0inv * pref * cov_perp
            self.results[
                "eps_perp_self"] = -beta * eps0inv * pref * cov_perp_self
            self.results[
                "eps_perp_coll"] = -beta * eps0inv * pref * cov_perp_coll
            self.results["deps_perp"] = np.abs(
                -eps0inv * beta * pref) * dcov_perp
            if (self.vac):
                self.results["eps_perp"] *= 2. / 3.
                self.results["eps_perp_self"] *= 2. / 3.
                self.results["eps_perp_coll"] *= 2. / 3.
                self.results["deps_perp"] *= 2. / 3.

        else:
            self.results["eps_perp"] = (- eps0inv * beta * pref * cov_perp) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)
            self.results["deps_perp"] = np.abs((- eps0inv * beta * pref) /
                                          (1 + eps0inv * beta * pref / self.results["V"] * var_perp)) * dcov_perp \
                + np.abs((- eps0inv * beta * pref * cov_perp) /
                         (1 + eps0inv * beta * pref / self.results["V"] * var_perp)**2) * dvar_perp

            self.results["eps_perp_self"] = (- eps0inv * beta * pref * cov_perp_self) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)
            self.results["eps_perp_coll"] = (- eps0inv * beta * pref * cov_perp_coll) \
                / (1 + eps0inv * beta * pref / self.results["V"] * var_perp)

    def _save_results(self):
        outdata_perp = np.hstack([
            self.results["z"][:, np.newaxis],
            self.results["eps_perp"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp"],
            np.linalg.norm(self.results["deps_perp"],
                           axis=1)[:, np.newaxis], self.results["deps_perp"],
            self.results["eps_perp_self"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp_coll"].sum(axis=1)[:, np.newaxis],
            self.results["eps_perp_self"], self.results["eps_perp_coll"]
        ])
        outdata_par = np.hstack([
            self.results["z"][:, np.newaxis],
            self.results["eps_par"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par"],
            np.linalg.norm(self.results["deps_par"],
                           axis=1)[:, np.newaxis], self.results["deps_par"],
            self.results["eps_par_self"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par_coll"].sum(axis=1)[:, np.newaxis],
            self.results["eps_par_self"], self.results["eps_par_coll"]
        ])

        # TODO: write general function to symmetrize lists
        # for include in more modules...
        if (self.bsym):
            for i in range(len(outdata_par) - 1):
                outdata_par[i + 1] = .5 * \
                    (outdata_par[i + 1] + outdata_par[i + 1][-1::-1])
            for i in range(len(outdata_perp) - 1):
                outdata_perp[i + 1] = .5 * \
                    (outdata_perp[i + 1] + outdata_perp[i + 1][-1::-1])

        header = "statistics over {:.1f} picoseconds".format(
            self._index * self._universe.trajectory.dt)
        savetxt("{}{}".format(self.output_prefix, "_perp"),
                outdata_perp,
                header=header)
        savetxt("{}{}".format(self.output_prefix, "_par"),
                outdata_par,
                header=header)


@charge_neutral(filter="error")
class epsilon_cylinder(SingleGroupAnalysisBase):
    """Calculate cylindrical dielectric profiles.
    
    Components are calculated along the axial (z) and radial (along xy) 
    direction at the system's center of mass.

    **Inputs**

    :param output_prefix (str): Prefix for output_prefix files
    :param geometry (str): A structure file without water from which com is calculated.
    :param radius (float): Radius of the cylinder (nm)
    :param binwidth (float): Bindiwdth the binwidth (nm)
    :param variable_dr (bool): Use a variable binwidth, where the volume is kept fixed.
    :param length (float): Length of the cylinder (nm)
    :param outfreq (int): Default number of frames after which output files are refreshed.
    :param temperature (float): temperature (K)
    :param single (bool): "1D" line of watermolecules
    :param bpbc (bool): Do not make broken molecules whole again (only works if
                        molecule is smaller than shortest box vector

    **Outputs**

    :returns (dict): * r: Bin positions
                     * eps_ax: Parallel dielectric profile (ε_∥)
                     * deps_ax: Error of parallel dielectric profile
                     * eps_rad: Inverse perpendicular dielectric profile (ε^{-1}_⟂)
                     * deps_rad: Error of inverse perpendicular dielectric profile
    """

    def __init__(self,
                 atomgroup,
                 output_prefix="eps_cyl",
                 binwidth=0.05,
                 outfreq=10000,
                 geometry=None,
                 radius=None,
                 variable_dr=False,
                 length=None,
                 temperature=300,
                 single=False,
                 bpbc=True,
                 **kwself):
        super().__init__(atomgroup, **kwself)
        self.output_prefix = output_prefix
        self.binwidth = binwidth
        self.outfreq = outfreq
        self.geometry = geometry
        self.radius = radius
        self.variable_dr = variable_dr
        self.length = length
        self.temperature = temperature
        self.single = single
        self.bpbc = bpbc

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output_prefix')
        parser.add_argument('-g', dest='geometry')
        parser.add_argument('-r', dest='radius')
        parser.add_argument('-dr', dest='binwidth')
        parser.add_argument('-vr', dest='variable_dr')
        parser.add_argument('-l', dest='length')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument('-si', dest='single')
        parser.add_argument('-nopbcrepair', dest='bpbc')

    def _prepare(self):

        if self.geometry is not None:
            self.com = self.system.atoms.center_of_mass(
                mda.Universe(self.geometry))
        else:
            if self._verbose:
                print("No geometry set."
                      " Calculate center of geometry from box dimensions.")
            self.com = self._universe.dimensions[:3] / 2

        if self.radius is not None:
            radius = 10 * self.radius
        else:
            if self._verbose:
                print("No radius set. Take smallest box extension.")
            radius = self._universe.dimensions[:2].min() / 2

        if self.length is not None:
            self.length *= 10
        else:
            self.length = self._universe.dimensions[2]

        # Convert nm -> Å
        self.binwidth *= 10

        self.nbins = int(np.ceil(radius / self.binwidth))

        if self.variable_dr:
            # variable dr
            sol = np.ones(self.nbins) * radius**2 / self.nbins
            mat = np.diag(np.ones(self.nbins)) + np.diag(
                np.ones(self.nbins - 1) * -1, k=-1)

            self.r_bins = np.sqrt(np.linalg.solve(mat, sol))
            self.dr = self.r_bins - np.insert(self.r_bins, 0, 0)[0:-1]
        else:
            # Constant dr
            self.dr = np.ones(self.nbins) * radius / self.nbins
            self.r_bins = np.arange(self.nbins) * self.dr + self.dr

        self.delta_r_sq = self.r_bins**2 - np.insert(self.r_bins, 0,
                                                     0)[0:-1]**2  # r_o^2-r_i^2
        self.r = np.copy(self.r_bins) - self.dr / 2

        # Use resampling for error estimation.
        # We do block averaging for 10 hardcoded blocks.
        self.resample = 10
        self.resample_freq = int(
            np.ceil((self.stopframe - self.startframe) / self.resample))

        self.m_rad = np.zeros((self.nbins, self.resample))

        self.M_rad = np.zeros((self.resample))
        self.mM_rad = np.zeros(
            (self.nbins, self.resample))  # total fluctuations

        self.m_ax = np.zeros((self.nbins, self.resample))
        self.M_ax = np.zeros((self.resample))
        self.mM_ax = np.zeros((self.nbins, self.resample))  # total fluctuations

        if self._verbose:
            print('Using', self.nbins, 'bins.')

    def _single_frame(self):

        if self.bpbc:
            # make broken molecules whole again!
            self._universe.atoms.unwrap(compound=check_compound(self._universe.atoms))

        # Transform from cartesian coordinates [x,y,z] to cylindrical
        # coordinates [r,z] (skip phi because of symmetry)
        positions_cyl = np.empty([self.atomgroup.positions.shape[0], 2])
        positions_cyl[:, 0] = np.linalg.norm(
            (self.atomgroup.positions[:, 0:2] - self.com[0:2]), axis=1)
        positions_cyl[:, 1] = self.atomgroup.positions[:, 2]

        # Use polarization density ( for radial component )
        # ========================================================
        bins_rad = np.digitize(positions_cyl[:, 0], self.r_bins)

        curQ_rad = np.histogram(bins_rad,
                                bins=np.arange(self.nbins + 1),
                                weights=self.atomgroup.charges)[0]
        this_m_rad = -np.cumsum(
            (curQ_rad / self.delta_r_sq) * self.r * self.dr) / (self.r * np.pi *
                                                                self.length)

        this_M_rad = np.sum(this_m_rad * self.dr)
        self.M_rad[self._frame_index // self.resample_freq] += this_M_rad

        self.m_rad[:, self._frame_index // self.resample_freq] += this_m_rad
        self.mM_rad[:, self._frame_index //
                    self.resample_freq] += this_m_rad * this_M_rad

        # Use virtual cutting method ( for axial component )
        # ========================================================
        nbinsz = 250  # number of virtual cuts ("many")

        this_M_ax = np.dot(self.atomgroup.charges, positions_cyl[:, 1])
        self.M_ax[self._frame_index // self.resample_freq] += this_M_ax

        # Move all r-positions to 'center of charge' such that we avoid monopoles in r-direction.
        # We only want to cut in z direction.
        chargepos = positions_cyl * np.abs(
            self.atomgroup.charges[:, np.newaxis])
        centers = self.atomgroup.accumulate(chargepos,
                                            compound=check_compound(self.atomgroup))
        centers /= self.atomgroup.accumulate(np.abs(self.atomgroup.charges),
                                             compound=check_compound(self.atomgroup))[:, np.newaxis]
        comp = check_compound(self.atomgroup)
        if comp == "molecules":
            repeats = np.unique(self.atomgroup.molnums, return_counts=True)[1]
        elif comp == "fragments":
            repeats = np.unique(self.atomgroup.fragindices, return_counts=True)[1]
        else:
            repeats = np.unique(self.atoms.resids, return_counts=True)[1]
        testpos = np.empty(positions_cyl[:, 0].shape)
        testpos = np.repeat(centers[:, 0], repeats)

        binsr = np.digitize(testpos, self.r_bins)

        dz = np.ones(nbinsz) * self.length / nbinsz
        z = np.arange(nbinsz) * dz + dz

        binsz = np.digitize(positions_cyl[:, 1], z)
        binsz[np.where(binsz < 0)] = 0
        curQz = np.histogram2d(
            binsr,
            binsz,
            bins=[np.arange(self.nbins + 1),
                  np.arange(nbinsz + 1)],
            weights=self.atomgroup.charges)[0]
        curqz = np.cumsum(curQz,
                          axis=1) / (np.pi * self.delta_r_sq)[:, np.newaxis]

        this_m_ax = -curqz.mean(axis=1)

        self.m_ax[:, self._frame_index // self.resample_freq] += this_m_ax
        self.mM_ax[:, self._frame_index //
                   self.resample_freq] += this_m_ax * this_M_ax

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1
        if self.single:  # removed average of M if single line water.
            cov_ax = self.mM_ax.sum(axis=1) / self._index
            cov_rad = self.mM_rad.sum(axis=1) / self._index

            dcov_ax = (self.mM_ax.std(axis=1) / self._index * self.resample) / \
                np.sqrt(self.resample - 1)
            dcov_rad = (self.mM_rad.std(axis=1) / self._index * self.resample) / \
                np.sqrt(self.resample - 1)
        else:
            cov_ax = self.mM_ax.sum(axis=1) / self._index - \
                self.m_ax.sum(axis=1) / self._index * self.M_ax.sum() / self._index
            cov_rad = self.mM_rad.sum(axis=1) / self._index - \
                self.m_rad.sum(axis=1) / self._index * self.M_rad.sum() / self._index

            dcov_ax = np.sqrt(
                (self.mM_ax.std(axis=1) / self._index * self.resample)**2 +
                (self.m_ax.std(axis=1) / self._index * self.resample *
                 self.M_ax.sum() / self._index)**2 +
                (self.m_ax.sum(axis=1) / self._index * self.M_ax.std() /
                 self._index * self.resample)**2) / np.sqrt(self.resample - 1)
            dcov_rad = np.sqrt(
                (self.mM_rad.std(axis=1) / self._index * self.resample)**2 +
                (self.m_rad.std(axis=1) / self._index * self.resample *
                 self.M_rad.sum() / self._index)**2 +
                (self.m_rad.sum(axis=1) / self._index * self.M_rad.std() /
                 self._index * self.resample)**2) / np.sqrt(self.resample - 1)

        beta = 1 / (scipy.constants.Boltzmann * self.temperature)

        self.results["eps_ax"] = 1 + beta * eps0inv * pref * cov_ax
        self.results["deps_ax"] = beta * eps0inv * pref * dcov_ax

        self.results[
            "eps_rad"] = 1 - beta * eps0inv * pref * 2 * np.pi * self.r * self.length * cov_rad
        self.results[
            "deps_rad"] = beta * eps0inv * pref * 2 * np.pi * self.r * self.length * dcov_rad

        self.results["r"] = self.r / 10

    def _save_results(self):

        outdata_ax = np.array([
            self.results["r"], self.results["eps_ax"], self.results["deps_ax"]
        ]).T
        outdata_rad = np.array([
            self.results["r"], self.results["eps_rad"], self.results["deps_rad"]
        ]).T

        header = "statistics over {:.1f} picoseconds".format(
            self._index * self._universe.trajectory.dt)
        savetxt("{}{}".format(self.output_prefix, "_ax.dat"),
                outdata_ax,
                header=header)
        savetxt("{}{}".format(self.output_prefix, "_rad.dat"),
                outdata_rad,
                header=header)


@charge_neutral(filter="error")
class dielectric_spectrum(SingleGroupAnalysisBase):
    """Computes the linear dielectric spectrum.

        This module, given molecular dynamics trajectory data, produces a
        .txt file containing the complex dielectric function as a function of the (linear, not radial -
        i.e. nu or f, rather than omega) frequency, along with the associated standard deviations.
        The algorithm is based on the Fluctuation Dissipation Relation (FDR):
        chi(f) = -1/(3 V k_B T epsilon_0) FT{theta(t) <P(0) dP(t)/dt>}.
        By default, the polarization trajectory, time series array and the average system volume are
        saved in the working directory, and the data are reloaded from these files if they are present.
        Lin-log and log-log plots of the susceptibility are also produced by default.

        **Inputs**

        :param recalc (bool): Forces to recalculate the polarization,
                              regardless if it is already present.
        :param temperature (float): Reference temperature.
        :param output_prefix (str): Prefix for the output files.
        :param segs (int): Sets the number of segments the trajectory is broken into.
        :param df (float): The desired frequency spacing in THz.
                           This determines the minimum frequency about which there
                           is data. Overrides -segs option.
        :param noplots (bool): Prevents plots from being generated.
        :param plotformat (bool): Allows the user to choose the format of generated plots.
        :param ymin (float): Manually sets the minimum lower bound for the log-log plot.
        :param bins (int): Determines the number of bins used for data averaging;
                           (this parameter sets the upper limit).
                           The data are by default binned logarithmically.
                           This helps to reduce noise, particularly in
                           the high-frequency domain, and also prevents plot
                           files from being too large.
        :param binafter (int): The number of low-frequency data points that are
                              left unbinned.
        :param nobin (bool): Prevents the data from being binned altogether. This
                             can result in very large plot files and errors.

        **Outputs**
        :returns (dict): TODO
    """

    # TODO set up script to calc spectrum at intervals while calculating polarization
    # for very big-data trajectories

    # TODO merge with molecular version?

    def __init__(self,
                 atomgroup,
                 recalc=False,
                 temperature=300,
                 output_prefix="",
                 segs=20,
                 df=None,
                 noplots=False,
                 plotformat="pdf",
                 ymin=None,
                 bins=200,
                 binafter=20,
                 nobin=False,
                 **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.temperature = temperature
        self.output_prefix = output_prefix
        self.segs = segs
        self.df = df
        self.noplots = noplots
        self.plotformat = plotformat
        self.ymin = ymin
        self.bins = bins
        self.binafter = binafter
        self.nobin = nobin

    def _configure_parser(self, parser):
        parser.add_argument("-recalc", dest="recalc")
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument("-o", dest="output_prefix")
        parser.add_argument("-segs", dest="segs")
        parser.add_argument("-df", dest="df")
        parser.add_argument("-noplots", dest="noplots")
        parser.add_argument("-plotformat", dest="plotformat")
        parser.add_argument("-ymin", dest="ymin")
        parser.add_argument("-bins", dest="bins")
        parser.add_argument("-binafter", dest="binafter")
        parser.add_argument("-nobin", dest="nobin")

    def _prepare(self):
        if self.plotformat not in ["pdf, png, jpg"]:
            raise ValueError(
                "Invalid choice for potformat: '{}' (choose from 'pdf', "
                "'png', 'jpg')".format(self.plotformat))

        if len(self.output_prefix) > 0:
            self.output_prefix += "_"

        self.Nframes = (self.stopframe - self.startframe) // self.step
        self.dt = self._trajectory.dt * self.step
        self.V = 0
        self.P = np.zeros((self.Nframes, 3))

    def _single_frame(self):
        self.V += self._ts.volume
        self.atomgroup.unwrap(compound=check_compound(self.atomgroup))
        self.P[self._frame_index, :] = np.dot(self.atomgroup.charges,
                                              self.atomgroup.positions)

    def _calculate_results(self):

        self.results["t"] = self._trajectory.dt * np.arange(
            self.startframe, self.stopframe, self.step)

        self.results["V"] = self.V
        self.results["V"] *= 1e-3 / (self._frame_index + 1)

        self.results["P"] = self.P
        # MDAnalysis gives units of Å, we use nm
        self.results["P"] /= 10

        # Find a suitable number of segments if it's not specified:
        if self.df is not None:
            self.segs = np.max([int(self.Nframes * self.dt * self.df), 2])

        self.seglen = int(self.Nframes / self.segs)

        # Prefactor for susceptibility:
        pref = scipy.constants.e * scipy.constants.e * 1e9 / \
            (3 * self.results["V"] * scipy.constants.k
             * self.temperature * scipy.constants.epsilon_0)

        if self._verbose:  # Susceptibility and errors:
            print('Calculating susceptibilty and errors...')

        # if t too short to simply truncate
        if len(self.results["t"]) < 2 * self.seglen:
            self.results["t"] = np.append(
                self.results["t"],
                self.results["t"] + self.results["t"][-1] + self.dt)

        # truncate t array (it's automatically longer than 2 * seglen)
        self.results["t"] = self.results["t"][:2 * self.seglen]
        # get freqs
        self.results["nu"] = FT(
            self.results["t"],
            np.append(self.results["P"][:self.seglen, 0],
                      np.zeros(self.seglen)))[0]
        # susceptibility
        self.results["susc"] = np.zeros(self.seglen, dtype=complex)
        # std deviation of susceptibility
        self.results["dsusc"] = np.zeros(self.seglen, dtype=complex)
        # susceptibility for current seg
        ss = np.zeros((2 * self.seglen), dtype=complex)

        # loop over segs
        for s in range(0, self.segs):
            if self._verbose:
                print('\rSegment {0} of {1}'.format(s + 1, self.segs), end='')
            ss = 0 + 0j

            # loop over x, y, z
            for self._i in range(0, len(self.results["P"][0, :])):
                FP = FT(
                    self.results["t"],
                    np.append(
                        self.results["P"][s * self.seglen:(s + 1) *
                                          self.seglen, self._i],
                        np.zeros(self.seglen)), False)
                ss += FP.real * FP.real + FP.imag * FP.imag

            ss *= self.results["nu"] * 1j

            # Get the real part by Kramers Kronig
            ss.real = iFT(
                self.results["t"], 1j * np.sign(self.results["nu"]) *
                FT(self.results["nu"], ss, False), False).imag

            if s == 0:
                self.results["susc"] += ss[self.seglen:]

            else:
                ds = ss[self.seglen:] - \
                    (self.results["susc"] / s)
                self.results["susc"] += ss[self.seglen:]
                dif = ss[self.seglen:] - \
                    (self.results["susc"] / (s + 1))
                ds.real *= dif.real
                ds.imag *= dif.imag
                # variance by Welford's Method
                self.results["dsusc"] += ds

        self.results["dsusc"].real = np.sqrt(self.results["dsusc"].real)
        self.results["dsusc"].imag = np.sqrt(self.results["dsusc"].imag)

        # 1/2 b/c it's the full FT, not only half-domain
        self.results["susc"] *= pref / (2 * self.seglen * self.segs * self.dt)
        self.results["dsusc"] *= pref / (2 * self.seglen * self.segs * self.dt)

        # Discard negative-frequency data; contains the same information as positive regime:
        # Now nu represents positive f instead of omega
        self.results["nu"] = self.results["nu"][self.seglen:] / (2 * np.pi)

        if self._verbose:
            print('Length of segments:    {0} frames, {1:.0f} ps'.format(
                self.seglen, self.seglen * self.dt))
            print('Frequency spacing:    ~ {0:.5f} THz'.format(
                self.segs / (self.Nframes * self.dt)))

        # Bin data if there are too many points:
        if not (self.nobin or self.seglen <= self.bins):
            bins = np.logspace(
                np.log(self.binafter) / np.log(10),
                np.log(len(self.results["susc"])) / np.log(10),
                self.bins - self.binafter + 1).astype(int)
            bins = np.unique(np.append(np.arange(self.binafter), bins))[:-1]

            self.results["nu_binned"] = Bin(self.results["nu"], bins)
            self.results["susc_binned"] = Bin(self.results["susc"], bins)
            self.results["dsusc_binned"] = Bin(self.results["dsusc"], bins)

            if self._verbose:
                print('Binning data above datapoint {0} in log-spaced bins'.
                      format(self.binafter))
                print('Binned data consists of {0} datapoints'.format(
                    len(self.results["susc"])))
        elif self._verbose:
            # data is binned
            print('Not binning data: there are {0} datapoints'.format(
                len(self.results["susc"])))

    def _save_results(self):
        np.save(self.output_prefix + 'tseries.npy', self.results["t"])

        with open(self.output_prefix + 'V.txt', "w") as Vfile:
            Vfile.write(str(self.results["V"]))

        np.save(self.output_prefix + 'P_tseries.npy', self.results["P"])

        suscfilename = "{}{}".format(self.output_prefix, 'susc')
        savetxt(
            suscfilename,
            np.transpose([
                self.results["nu"], self.results["susc"].real,
                self.results["dsusc"].real, self.results["susc"].imag,
                self.results["dsusc"].imag
            ]),
            delimiter='\t',
            header='freq\tsusc\'\tstd_dev_susc\'\t-susc\'\'\tstd_dev_susc\'\'')

        if self._verbose:
            print('Susceptibility data saved as ' + suscfilename)

        if not (self.nobin or self.seglen <= self.bins):

            suscfilename = "{}{}".format(self.output_prefix, 'susc_binned')
            savetxt(suscfilename,
                    np.transpose([
                        self.results["nu_binned"],
                        self.results["susc_binned"].real,
                        self.results["dsusc_binned"].real,
                        self.results["susc_binned"].imag,
                        self.results["dsusc_binned"].imag
                    ]),
                    delimiter='\t',
                    header="freq\tsusc\'\tstd_dev_susc\'\t-"
                    "susc\'\'\tstd_dev_susc\'\'")

            if self._verbose:
                print('Binned susceptibility data saved as ' + suscfilename)

    def _conclude(self):
        if self.noplots and self._verbose:
            print('User specified not to generate plots -- finished :)')

        else:
            if self._verbose:
                print('Generating plots...')

            import matplotlib.pyplot as plt

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')

            # Colors/alpha values/labels/params for plotting
            col1 = 'royalblue'
            col2 = 'crimson'
            curve = 0.9
            shade = 0.15
            lw = 1.0
            nuBuf = 1.4  # buffer factor for extra room in the x direction
            cp = r'$\chi^{{\prime}}$'
            cpp = r'$\chi^{{\prime \prime}}$'
            width = 3.5  # inches

            def my_plot(binned=False):
                element = binned * "_binned"

                fig, ax = plt.subplots(1, figsize=[width, width / np.sqrt(2)])
                ax.set_ylabel(r'$\chi$')
                ax.set_xlabel('$\\nu$ [THz]')
                ax.set_xlim(self.results["nu"][1] / nuBuf,
                            self.results["nu"][-1] * nuBuf)
                ax.set_xscale('log')
                ax.set_yscale(yscale)

                ax.fill_between(self.results["nu" + element][1:],
                                self.results["susc" + element].real[1:] -
                                self.results["dsusc" + element].real[1:],
                                self.results["susc" + element].real[1:] +
                                self.results["dsusc" + element].real[1:],
                                color=col2,
                                alpha=shade)
                ax.fill_between(self.results["nu" + element][1:],
                                self.results["susc" + element].imag[1:] -
                                self.results["dsusc" + element].imag[1:],
                                self.results["susc" + element].imag[1:] +
                                self.results["dsusc" + element].imag[1:],
                                color=col1,
                                alpha=shade)

                ax.plot(self.results["nu" + element][:2],
                        self.results["susc" + element].real[:2],
                        color=col2,
                        alpha=curve,
                        linestyle=':',
                        linewidth=lw)
                ax.plot(self.results["nu" + element][:2],
                        self.results["susc" + element].imag[:2],
                        color=col1,
                        alpha=curve,
                        linestyle=':',
                        linewidth=lw)
                ax.plot(self.results["nu" + element][1:],
                        self.results["susc" + element].real[1:],
                        color=col2,
                        alpha=curve,
                        label=cp,
                        linewidth=lw)
                ax.plot(self.results["nu" + element][1:],
                        self.results["susc" + element].imag[1:],
                        color=col1,
                        alpha=curve,
                        label=cpp,
                        linewidth=lw)

                if self._i == 0 and (self.ymin is not None):
                    plt.set_ylim(ymin=self.ymin)
                ax.legend(loc='best', frameon=False)
                fig.tight_layout(pad=0.1)
                fig.savefig(plotname, format=self.plotformat)

            if self.nobin or self.seglen <= self.bins:
                binned = False
            else:
                binned = True

            yscale = 'log'
            plotname = self.output_prefix + 'susc_log.' + self.plotformat
            my_plot(binned)  # log-log

            yscale = 'linear'
            plotname = self.output_prefix + 'susc_linlog.' + self.plotformat
            my_plot(binned)  # lin-log

            plt.close('all')

            if self._verbose:
                print('Susceptibility plots generated -- finished :)')
