#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tool for computing transport properties.

The transport module of MAICoS allows for calculating mean velocity
profiles from molecular simulation trajectory files.
"""

import numpy as np
from scipy.optimize import curve_fit

from ..decorators import make_whole, set_verbose_doc
from ..utils import check_compound, savetxt
from .base import AnalysisBase


def fitfn(x, alpha, tau1, tau2, pref):
    """Fit fn."""
    return pref * (alpha * tau1 * (1 + tau1 / x * (np.exp(-x / tau1) - 1))
                   + (1 - alpha) * tau2 * (1 + tau2 / x
                                           * (np.exp(-x / tau2) - 1)))


@set_verbose_doc
@make_whole()
class Velocity(AnalysisBase):
    """Analyse mean velocity.

    Reads in coordinates and velocities from a trajectory and calculates a
    velocity profile along a given axis. The obtained profile is averaged
    over the 4 symmetric slab halfs. Error bars are estimated via block
    averaging as described in Ref. [1].

    [1] Hess, B. Determining the shear viscosity of model liquids from
    molecular dynamics simulations. The Journal of Chemical Physics
    116, 209-217 (2002).

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    dim : int
        Dimension for position binning (x=0, y=1, z=2)
    vdim : int
        Dimension for velocity binning (x=0, y=1, z=2)
    n_bins : int
        Number of bins.
        For making use of symmetry must be a multiple of 4.
    n_block : int
        Maximum number of blocks for block averaging error estimate;
        1 results in standard error
    ${MAKE_WHOLE_PARAMETER}
    output_suffix : str
        Suffix for output filenames
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at
        the end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.z : np.ndarray
        bins [nm]
    results.v : np.ndarray
        velocity profile [Å/ps]
    results.ees : np.ndarray
        velocity error estimate [Å/ps]
    results.symz : np.ndarray
        symmetrized bins [Å/ps]
    results.symvel : np.ndarray
        symmetrized velocity profile [Å/ps]
    results.symees : np.ndarray
        symmetrized velocity error estimate [Å/ps]
    """

    def __init__(self,
                 atomgroup,
                 dim=2,
                 vdim=0,
                 n_bins=200,
                 n_block=10,
                 make_whole=True,
                 output_suffix="com",
                 concfreq=0,
                 **kwargs):
        super(Velocity, self).__init__(atomgroup, **kwargs)
        self.dim = dim
        self.vdim = vdim
        self.n_bins = n_bins
        self.n_block = n_block
        self.make_whole = make_whole
        self.output_suffix = output_suffix
        self.concfreq = concfreq

    def _prepare(self):

        if self.n_bins % 2 != 0:
            raise ValueError("Number of bins %d can't be divided by 4!")

        self.blockfreq = int(np.ceil(self.n_frames / self.n_block))
        # skip from initial, not end
        self.skipinitialframes = self.n_frames % self.n_block

        self.av_vel = np.zeros((self.n_bins, self.n_block))
        self.av_vel_sq = np.zeros((self.n_bins))
        # count frame only to velocity if existing
        self.binframes = np.zeros((self.n_bins, self.n_block))
        self.L = 0

    def _single_frame(self):
        self.L += self._universe.dimensions[self.dim]

        coms = self.atomgroup.center_of_mass(
            compound=check_compound(self.atomgroup))[:, self.dim]

        comvels = self.atomgroup.atoms.accumulate(
            self.atomgroup.atoms.velocities[:, self.vdim]
            * self.atomgroup.atoms.masses,
            compound=check_compound(self.atomgroup)
            )
        comvels /= self.atomgroup.atoms.accumulate(self.atomgroup.atoms.masses,
                                                   compound=check_compound(
                                                       self.atomgroup))

        curvel = np.histogram(coms,
                              bins=self.n_bins,
                              range=(0, self._ts.dimensions[self.dim]),
                              weights=comvels)[0]
        bincount = np.histogram(coms,
                                bins=self.n_bins,
                                range=(0, self._ts.dimensions[self.dim]))[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            # mean velocity in this bin, zero if empty
            curvel /= bincount
        curvel = np.nan_to_num(curvel)

        # add velocities to the average
        self.av_vel[:, self._frame_index // self.blockfreq] += curvel
        self.av_vel_sq[:] += (curvel)**2
        # only average velocities if bin is not empty
        self.binframes[:, self._frame_index // self.blockfreq] += bincount > 0

    def _conclude(self):
        """Calculate the results."""
        # minimum number of frames where molecules should be present
        self.minframes = self._index / 100
        avL = self.L / self._index
        dz = avL / self.n_bins
        self.results.symz = np.arange(0, avL / 4 - dz / 2, dz) + dz / 2

        self.results.z = np.arange(0, avL - dz / 2, dz) + dz / 2
        self.results.v = np.sum(
            self.av_vel[np.sum(self.binframes, axis=1) > self.minframes],
            axis=1) / np.sum(
                self.binframes[np.sum(self.binframes, axis=1)
                               > self.minframes],
                axis=1)
        self.results.dv = np.sqrt(self.av_vel_sq[
            np.sum(self.binframes, axis=1) > self.minframes] / np.sum(
                self.binframes[np.sum(self.binframes, axis=1)
                               > self.minframes],
                axis=1) - self.results.v**2) / np.sqrt(
                    np.sum(self.binframes[
                        np.sum(self.binframes, axis=1) > self.minframes],
                        axis=1) - 1)

        # make use of the symmetry
        self.results.symvel = (
            self.av_vel[:self.n_bins // 4]
            - self.av_vel[self.n_bins // 4:2 * self.n_bins // 4][::-1]
            - self.av_vel[2 * self.n_bins // 4:3 * self.n_bins // 4]
            + self.av_vel[3 * self.n_bins // 4:][::-1])
        self.results.symvel_sq = (
            self.av_vel_sq[:self.n_bins // 4]
            + self.av_vel_sq[self.n_bins // 4:2 * self.n_bins // 4][::-1]
            + self.av_vel_sq[2 * self.n_bins // 4:3 * self.n_bins // 4]
            + self.av_vel_sq[3 * self.n_bins // 4:][::-1])
        self.results.symbinframes = (
            self.binframes[:self.n_bins // 4]
            + self.binframes[self.n_bins // 4:2 * self.n_bins // 4][::-1]
            + self.binframes[2 * self.n_bins // 4:3 * self.n_bins // 4]
            + self.binframes[3 * self.n_bins // 4:][::-1])

        self.results.vsym = np.sum(
            self.results.symvel[
                np.sum(self.results.symbinframes, axis=1) > self.minframes],
            axis=1,
            ) / np.sum(
            self.results.symbinframes[
                np.sum(self.results.symbinframes, axis=1) > self.minframes],
            axis=1,
            )
        self.results.dvsym = np.sqrt(self.results.symvel_sq[np.sum(
            self.results.symbinframes, axis=1) > self.minframes] / np.sum(
                self.results.symbinframes
                [np.sum(self.results.symbinframes, axis=1) > self.minframes],
                axis=1,
                ) - self.results.vsym**2) / np.sqrt(
            np.sum(
                self.results.symbinframes[np.sum(
                    self.results.symbinframes, axis=1) > self.minframes],
                axis=1,
                ) - 1)

        bee = self._blockee(np.nan_to_num(self.av_vel / self.binframes))
        self.results.ee_out = np.vstack(
            list(np.hstack((bee[i])) for i in range(len(bee))))

        # 2 sigma^2 / T, (A16) in [1]
        prefs = (2 * (self.av_vel_sq[np.sum(self.binframes, axis=1)
                                     > self.minframes]
                      / np.sum(self.binframes[np.sum(self.binframes, axis=1)
                                              > self.minframes], axis=1,)
                      - self.results.v**2)
                 / (self._index * self._trajectory.dt * self.step))
        self.results.ees = []
        self.results.params = []
        for count, i in enumerate(range(self.n_bins)):
            if np.sum(self.binframes[i]) > self.minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2, pref=pref):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    self.results.ee_out[:, 0],
                    (self.results.ee_out[:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[0.99, 0.001, 0.01],
                    max_nfev=1e5,
                    )
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                self.results.ees.append(errest)
                self.results.params.append([pref, alpha, tau1, tau2])

        # Same for symmetrized
        bee = self._blockee(
            np.nan_to_num(self.results.symvel
                          / self.results.symbinframes))
        self.results.symee_out = np.vstack(
            list(np.hstack((bee[i])) for i in range(len(bee))))

        # 2 sigma^2 / T, (A16) in [1]
        prefs = (2 * (self.results.symvel_sq
                      [np.sum(self.results.symbinframes, axis=1)
                       > self.minframes]
                      / np.sum(self.results.symbinframes
                               [np.sum(self.results.symbinframes, axis=1)
                                > self.minframes],
                               axis=1,)
                      - self.results.vsym**2)
                 / (self._index * self._trajectory.dt * self.step))

        self.results.symees = []
        for count, i in enumerate(range(self.n_bins // 4)):
            if np.sum(self.results.symbinframes[i]) > self.minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2, pref=pref):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    self.results.symee_out[:, 0],
                    (self.results.symee_out[:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[0.9, 1e3, 1e4],
                    max_nfev=1e5,
                    )
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                self.results.symees.append(errest)

    def _blockee(self, data):
        ee = []
        for i in range(0, int(np.log2(self.n_block)) - 1):
            bs = 2**i
            numb = self.n_block // bs
            blocks = np.vstack([
                np.mean(data[:, bs * i:bs * (i + 1)], axis=1)
                for i in range(numb)
                ]).T
            ee.append([
                bs * self._trajectory.dt * self.step * self.blockfreq,
                np.std(blocks, axis=1) / np.sqrt(numb - 1),
                ])
        return ee

    def save(self):
        """Save result."""
        savetxt("errest_" + self.output_suffix,
                np.concatenate((self.results.ee_out[:, 0].
                                reshape(len(self.results.ee_out), 1),
                                (self.results.ee_out[:, 1:])
                                [:, np.sum(self.binframes, axis=1)
                                > self.minframes],), axis=1,),
                header="z " + " ".join(map(str,
                                           self.results.z[
                                               np.sum(self.binframes, axis=1)
                                               > self.minframes],)),)
        savetxt("errparams_" + self.output_suffix,
                np.array(self.results.params))
        savetxt(
            "errest_sym_" + self.output_suffix,
            np.concatenate(
                (self.results.symee_out[:, 0].reshape(
                    len(self.results.symee_out), 1),
                 (self.results.symee_out[:, 1:])[:,
                 np.sum(self.results.symbinframes, axis=1)
                 > self.minframes],), axis=1,),
            header="z " + " ".join(
                map(
                    str,
                    self.results.symz
                    [np.sum(self.results.symbinframes, axis=1)
                     > self.minframes],
                    )),
            )
        savetxt(
            "errparams_sym_" + self.output_suffix,
            np.array(self.results.params),
            )

        savetxt(
            "vel_" + self.output_suffix,
            np.vstack((
                self.results.z[
                    np.sum(self.binframes, axis=1) > self.minframes],
                self.results.v,
                np.array(self.results.ees),
                self.results.dv,
                )).T,
            )

        savetxt(
            "vel_sym_" + self.output_suffix,
            np.vstack((
                self.results.symz[np.sum(self.results.symbinframes,
                                         axis=1) > self.minframes],
                self.results.vsym,
                np.array(self.results.symees),
                )).T,
            )
