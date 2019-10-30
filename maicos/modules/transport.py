#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
from scipy.optimize import curve_fit

from .base import SingleGroupAnalysisBase
from ..utils import savetxt


def fitfn(x, alpha, tau1, tau2, pref):
    return pref * (alpha * tau1 * (1 + tau1 / x * (np.exp(-x / tau1) - 1)) +
                   (1 - alpha) * tau2 * (1 + tau2 / x *
                                         (np.exp(-x / tau2) - 1)))


class velocity(SingleGroupAnalysisBase):
    """Mean Velocity analysis.

        Reads in coordinates and velocities from a trajectory and calculates a
        velocity profile along a given axis. The obtained profile is averaged over the 4
        symmetric slab halfs.Error bars are estimated via block averaging as described in [1].

        [1] Hess, B. Determining the shear viscosity of model liquids from molecular
           dynamics simulations. The Journal of Chemical Physics 116, 209-217 (2002).

        :param output_suffix (str): Suffix for output filenames
        :param outfreq (int): Default time after which output files are refreshed (1000 ps).
        :param dim (int): Dimension for position binning (0=X, 1=Y, 2=Z)
        :param vdim (int): Dimension for velocity binning (0=X, 1=Y, 2=Z)
        :param nbins (int): Number of bins.
                            For making use of symmetry must be a multiple of 4.
        :param nblock (int): Maximum number of blocks for block averaging error estimate;
                            1 results in standard error
        :param bpbc (bool): Do not make broken molecules whole again (only works if
                            molecule is smaller than shortest box vector

        :returns (dict): * z: bins [nm]
                         * v: velocity profile [m/s]
                         * ees: velocity error estimate [m/s]
                         * symz: symmetrized bins [nm]
                         * symvel: symmetrized velocity profile [m/s]
                         * symees: symmetrized velocity error estimate [m/s]
     """

    def __init__(self,
                 atomgroups,
                 output_suffix="com",
                 outfreq=1000,
                 dim=2,
                 vdim=0,
                 nbins=200,
                 nblock=10,
                 bpbc=True,
                 **kwargs):
        super(velocity, self).__init__(atomgroups, **kwargs)
        self.output_suffix = output_suffix
        self.outfreq = outfreq
        self.dim = dim
        self.vdim = vdim
        self.nbins = nbins
        self.nblock = nblock
        self.bpbc = bpbc

    def _configure_parser(self, parser):
        parser.add_argument("-o", dest="output_suffix")
        parser.add_argument("-dout", dest="outfreq")
        parser.add_argument("-d", dest="dim")
        parser.add_argument("-dv", dest="vdim")
        parser.add_argument("-nbins", dest="nbins")
        parser.add_argument("-nblock", dest="nblock")
        parser.add_argument("-nopbcrepair", dest="bpbc")

    def _prepare(self):

        if self.nbins % 2 != 0:
            raise ValueError("Number of bins %d can't be divided by 4!")

        self.blockfreq = int(
            np.ceil((self.stopframe - self.startframe) / self.nblock))
        # skip from initial, not end
        self.skipinitialframes = self.n_frames % self.nblock

        self.av_vel = np.zeros((self.nbins, self.nblock))
        self.av_vel_sq = np.zeros((self.nbins))
        # count frame only to velocity if existing
        self.binframes = np.zeros((self.nbins, self.nblock))
        self.L = 0

    def _single_frame(self):
        if self.bpbc:
            # make broken molecules whole again!
            self._universe.atoms.unwrap(compound="molecules")

        self.L += self._universe.dimensions[self.dim]

        coms = self.atomgroup.center_of_mass(compound="molecules")[:, self.dim]

        comvels = self.atomgroup.atoms.accumulate(
            self.atomgroup.atoms.velocities[:, self.vdim] *
            self.atomgroup.atoms.masses,
            compound="molecules",
        )

        bins = (coms / (self._universe.dimensions[self.dim] / self.nbins)
               ).astype(int) % self.nbins
        bincount = np.bincount(bins, minlength=self.nbins)
        with np.errstate(divide="ignore", invalid="ignore"):
            # mean velocity in this bin, zero if empty
            curvel = np.nan_to_num(
                np.histogram(bins,
                             bins=np.arange(0, self.nbins + 1),
                             weights=comvels)[0] / bincount)

        # add velocities to the average and convert to (m/s)
        self.av_vel[:, self._frame_index // self.blockfreq] += curvel * 100
        self.av_vel_sq[:] += (curvel * 100)**2
        # only average velocities if bin is not empty
        self.binframes[:, self._frame_index // self.blockfreq] += bincount > 0

        if (self._save and self._frame_index % self.outfreq == 0 and
                self._frame_index > 0):
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        """Calculate the results."""

        self._index = self._frame_index + 1

        # minimum number of frames where molecules should be present
        self.minframes = self._index / 100
        avL = self.L / self._index / 10  # in nm
        dz = avL / self.nbins
        self.results["symz"] = np.arange(0, avL / 4 - dz / 2, dz) + dz / 2

        self.results["z"] = np.arange(0, avL - dz / 2, dz) + dz / 2
        self.results["v"] = np.sum(
            self.av_vel[np.sum(self.binframes, axis=1) > self.minframes],
            axis=1) / np.sum(
                self.binframes[np.sum(self.binframes, axis=1) > self.minframes],
                axis=1)
        self.results["dv"] = np.sqrt(self.av_vel_sq[
            np.sum(self.binframes, axis=1) > self.minframes] / np.sum(
                self.binframes[np.sum(self.binframes, axis=1) > self.minframes],
                axis=1) - self.results["v"]**2) / np.sqrt(
                    np.sum(self.binframes[
                        np.sum(self.binframes, axis=1) > self.minframes],
                           axis=1) - 1)

        # make use of the symmetry
        self.results["symvel"] = (
            self.av_vel[:self.nbins // 4] -
            self.av_vel[self.nbins // 4:2 * self.nbins // 4][::-1] -
            self.av_vel[2 * self.nbins // 4:3 * self.nbins // 4] +
            self.av_vel[3 * self.nbins // 4:][::-1])
        self.results["symvel_sq"] = (
            self.av_vel_sq[:self.nbins // 4] +
            self.av_vel_sq[self.nbins // 4:2 * self.nbins // 4][::-1] +
            self.av_vel_sq[2 * self.nbins // 4:3 * self.nbins // 4] +
            self.av_vel_sq[3 * self.nbins // 4:][::-1])
        self.results["symbinframes"] = (
            self.binframes[:self.nbins // 4] +
            self.binframes[self.nbins // 4:2 * self.nbins // 4][::-1] +
            self.binframes[2 * self.nbins // 4:3 * self.nbins // 4] +
            self.binframes[3 * self.nbins // 4:][::-1])

        self.results["vsym"] = np.sum(
            self.results["symvel"][
                np.sum(self.results["symbinframes"], axis=1) > self.minframes],
            axis=1,
        ) / np.sum(
            self.results["symbinframes"][
                np.sum(self.results["symbinframes"], axis=1) > self.minframes],
            axis=1,
        )
        self.results["dvsym"] = np.sqrt(self.results["symvel_sq"][np.sum(
            self.results["symbinframes"], axis=1) > self.minframes] / np.sum(
                self.results["symbinframes"]
                [np.sum(self.results["symbinframes"], axis=1) > self.minframes],
                axis=1,
            ) - self.results["vsym"]**2) / np.sqrt(
                np.sum(
                    self.results["symbinframes"][np.sum(
                        self.results["symbinframes"], axis=1) > self.minframes],
                    axis=1,
                ) - 1)

    def _blockee(self, data):
        ee = []
        for i in range(0, int(np.log2(self.nblock)) - 1):
            bs = 2**i
            numb = self.nblock // bs
            blocks = np.vstack([
                np.mean(data[:, bs * i:bs * (i + 1)], axis=1)
                for i in range(numb)
            ]).T
            ee.append([
                bs * self._trajectory.dt * self.step * self.blockfreq,
                np.std(blocks, axis=1) / np.sqrt(numb - 1),
            ])
        return ee

    def _conclude(self):
        bee = self._blockee(np.nan_to_num(self.av_vel / self.binframes))
        self.results["ee_out"] = np.vstack(
            list(np.hstack((bee[i])) for i in range(len(bee))))

        prefs = (2 * (
            self.av_vel_sq[np.sum(self.binframes, axis=1) > self.minframes] /
            np.sum(
                self.binframes[np.sum(self.binframes, axis=1) > self.minframes],
                axis=1,
            ) - self.results["v"]**2) /
                 (self._index * self._trajectory.dt * self.step)
                )  # 2 sigma^2 / T, (A16) in [1]
        self.results["ees"] = []
        self.results["params"] = []
        for count, i in enumerate(range(self.nbins)):
            if np.sum(self.binframes[i]) > self.minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    self.results["ee_out"][:, 0],
                    (self.results["ee_out"][:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[0.99, 0.001, 0.01],
                    max_nfev=1e5,
                )
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                self.results["ees"].append(errest)
                self.results["params"].append([pref, alpha, tau1, tau2])

        # Same for symmetrized
        bee = self._blockee(
            np.nan_to_num(self.results["symvel"] /
                          self.results["symbinframes"]))
        self.results["symee_out"] = np.vstack(
            list(np.hstack((bee[i])) for i in range(len(bee))))

        prefs = (2 * (self.results["symvel_sq"][np.sum(
            self.results["symbinframes"], axis=1) > self.minframes] / np.sum(
                self.results["symbinframes"]
                [np.sum(self.results["symbinframes"], axis=1) > self.minframes],
                axis=1,
            ) - self.results["vsym"]**2) /
                 (self._index * self._trajectory.dt * self.step)
                )  # 2 sigma^2 / T, (A16) in [1]
        self.results["symees"] = []
        for count, i in enumerate(range(self.nbins // 4)):
            if np.sum(self.results["symbinframes"][i]) > self.minframes:
                pref = prefs[count]

                def modfitfn(x, alpha, tau1, tau2):
                    return fitfn(x, alpha, tau1, tau2, pref)

                [alpha, tau1, tau2], pcov = curve_fit(
                    modfitfn,
                    self.results["symee_out"][:, 0],
                    (self.results["symee_out"][:, i + 1])**2,
                    bounds=([0, 0, 0], [1, np.inf, np.inf]),
                    p0=[0.9, 1e3, 1e4],
                    max_nfev=1e5,
                )
                # (A.17) in [1]
                errest = np.sqrt(pref * (alpha * tau1 + (1 - alpha) * tau2))
                self.results["symees"].append(errest)

        if self._save:
            savetxt(
                "errest_" + self.output_suffix,
                np.concatenate(
                    (
                        self.results["ee_out"][:, 0].reshape(
                            len(self.results["ee_out"]), 1),
                        (self.results["ee_out"][:, 1:]
                        )[:, np.sum(self.binframes, axis=1) > self.minframes],
                    ),
                    axis=1,
                ),
                header="z " + " ".join(
                    map(
                        str,
                        self.results["z"][
                            np.sum(self.binframes, axis=1) > self.minframes],
                    )),
            )
            savetxt("errparams_" + self.output_suffix,
                    np.array(self.results["params"]))
            savetxt(
                "errest_sym_" + self.output_suffix,
                np.concatenate(
                    (
                        self.results["symee_out"][:, 0].reshape(
                            len(self.results["symee_out"]), 1),
                        (self.results["symee_out"][:, 1:]
                        )[:,
                          np.sum(self.results["symbinframes"], axis=1) > self.
                          minframes],
                    ),
                    axis=1,
                ),
                header="z " + " ".join(
                    map(
                        str,
                        self.results["symz"]
                        [np.sum(self.results["symbinframes"], axis=1) >
                         self.minframes],
                    )),
            )
            savetxt(
                "errparams_sym_" + self.output_suffix,
                np.array(self.results["params"]),
            )

            savetxt(
                "vel_" + self.output_suffix,
                np.vstack((
                    self.results["z"][
                        np.sum(self.binframes, axis=1) > self.minframes],
                    self.results["v"],
                    np.array(self.results["ees"]),
                    self.results["dv"],
                )).T,
            )

            savetxt(
                "vel_sym_" + self.output_suffix,
                np.vstack((
                    self.results["symz"][np.sum(self.results["symbinframes"],
                                                axis=1) > self.minframes],
                    self.results["vsym"],
                    np.array(self.results["symees"]),
                )).T,
            )

    def _save_results(self):
        savetxt(
            "vel_sym_" + self.output_suffix,
            np.vstack((
                self.results["symz"]
                [np.sum(self.results["symbinframes"], axis=1) > self.minframes],
                self.results["vsym"],
                self.results["dvsym"],
            )).T,
        )
        savetxt(
            "vel_" + self.output_suffix,
            np.vstack((
                self.results["z"][
                    np.sum(self.binframes, axis=1) > self.minframes],
                self.results["v"],
                self.results["dv"],
            )).T,
        )
