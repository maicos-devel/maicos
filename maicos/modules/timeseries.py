#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np

from ..lib.utils import check_compound, savetxt
from .base import SingleGroupAnalysisBase, MultiGroupAnalysisBase


class dipole_angle(SingleGroupAnalysisBase):
    """Calculates the timeseries of the dipole moment with respect to an axis.

    :param dim (int): refernce vector for angle (x,y,z=0,1,2)
    :param outfreq (float): Default number of frames after which output files are refreshed
    :param output (str): Prefix for output filenames

    :returns (dict): * t: time (ps)
                     * cos_theta_i: Average cos between dipole and axis
                     * cos_theta_ii: Average cos^2 of the same between dipole and axis
                     * cos_theta_ij: Product cos of dipole i and cos of dipole j (i!=j)
    """

    def __init__(self,
                 atomgroup,
                 output="dipangle.dat",
                 outfreq=10000,
                 dim=2,
                 **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.output = output
        self.dim = dim
        self.outfreq = outfreq

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument('-d', dest='dim')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-o', dest='output')

    def _prepare(self):
        self.n_residues = self.atomgroup.residues.n_residues

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self):

        # make broken molecules whole again!
        self.atomgroup.unwrap(compound="molecule")

        chargepos = self.positions * self.charges[:, np.newaxis]
        dipoles = self.atomgroup.accumulate(chargepos, compound=check_compound(self.atomgroup))

        cos_theta = np.dot(dipoles, self.unit) / \
            np.linalg.norm(dipoles, axis=1)
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        self.cos_theta_i[self._frame_index] = cos_theta.mean()
        self.cos_theta_ii[self._frame_index] = trace / self.n_residues
        self.cos_theta_ij[self._frame_index] = (matrix.sum() - trace)
        self.cos_theta_ij[self._frame_index] /= (self.n_residues**2 -
                                                 self.n_residues)

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["t"] = self._trajectory.dt * \
            np.arange(self.startframe, self.stopframe, self.step)

        self.results["cos_theta_i"] = self.cos_theta_i[:self._index]
        self.results["cos_theta_ii"] = self.cos_theta_ii[:self._index]
        self.results["cos_theta_ij"] = self.cos_theta_ij[:self._index]

    def _save_results(self):

        savetxt(self.output,
                np.vstack([
                    self.results["t"], self.results["cos_theta_i"],
                    self.results["cos_theta_ii"], self.results["cos_theta_ij"]
                ]).T,
                header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
                fmt='%.5e')


class kinetic_energy(SingleGroupAnalysisBase):
    """Calculates the timeseries for the molecular center
       translational and rotational kinetic energy (kJ/mole).

       :param output (str): Output filename
       :param refpoint (str): reference point for molecular center: center of
                              mass (COM), center of charge (COC), or oxygen position (OXY)
                              Note: The oxygen position only works for systems of pure water

        :returns (dict): * t: time (ps)
                         * trans: translational kinetic energy (kJ/mole)
                         * rot: rotational kinetic energy (kJ/mole)
        """

    def __init__(self, atomgroup, output="ke.dat", refpoint="COM", **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.output = output
        self.refpoint = refpoint

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-r', dest='refpoint')

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        if self.refpoint not in ["COM", "COC", "OXY"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'COM', "
                "'COC', 'OXY')".format(self.refpoint))

        if self.refpoint == "OXY":
            self.oxy = self.atomgroup.select_atoms("name OW*")

        self.masses = self.atomgroup.atoms.accumulate(
            self.atomgroup.atoms.masses, compound=check_compound(self.atomgroup))
        self.abscharges = self.atomgroup.atoms.accumulate(np.abs(
            self.atomgroup.atoms.charges), compound=check_compound(self.atomgroup))
        # Total kinetic energy
        self.E_kin = np.zeros(self.n_frames)

        # Molecular center energy
        self.E_center = np.zeros(self.n_frames)

    def _single_frame(self):
        self.E_kin[self._frame_index] = np.dot(
            self.atomgroup.masses,
            np.linalg.norm(self.atomgroup.velocities, axis=1)**2)

        if self.refpoint == "COM":
            massvel = self.atomgroup.velocities * \
                self.atomgroup.masses[:, np.newaxis]
            v = self.atomgroup.accumulate(massvel, compound=check_compound(self.atomgroup))
            v /= self.masses[:, np.newaxis]

        elif self.refpoint == "COC":
            abschargevel = self.atomgroup.velocities * \
                np.abs(self.atomgroup.charges)[:, np.newaxis]
            v = self.atomgroup.accumulate(abschargevel, compound=check_compound(self.atomgroup))
            v /= self.abscharges[:, np.newaxis]

        elif self.refpoint == "OXY":
            v = self.oxy.velocities

        self.E_center[self._frame_index] = np.dot(self.masses,
                                                  np.linalg.norm(v, axis=1)**2)

    def _calculate_results(self):
        self.results["t"] = self._trajectory.dt * \
            np.arange(self.startframe, self.stopframe, self.step)
        self.results["trans"] = self.E_center / 2 / 100
        self.results["rot"] = (self.E_kin - self.E_center) / 2 / 100

    def _save_results(self):
        savetxt(self.output,
                np.vstack([
                    self.results["t"], self.results["trans"],
                    self.results["rot"]
                ]).T,
                fmt='%.8e',
                header="t / ps \t E_kin^trans / kJ/mole \t E_kin^rot / kJ/mole")


class dipole_trajectory(MultiGroupAnalysisBase):
    """
    Calculates the dipole and charge current trajectories for a series of atomgroups.

   :param output_prefix (str): Prefix for the output files.
   :param restypes (str): Types of residues contained in each atomgroup.
                            Options are "SP" for a single particle, "NM" for a neutral molecule,
                            or "CM" for a generic, charged molecule.
                            If not supplied, "CM" is assumed.
                            Affects the method of calculating each dipole component,
                            wrongly choosing may result in incorrect results.
    :param labels (str): Labels for each atomgroup.
    :param nojump (bool): Indicates atomgroups are unfolded across boundaries,
                            and the translational dipole moment can be computed comtinuously.
    :param bpbc (bool): Do not make broken molecules whole again (only works if
                        molecule is smaller than shortest box vector

    :returns (dict): * time: time (ps)
                     * LABEL_MD: Rotational dipole moment of group
                     * LABEL_MJ: Translational dipole moment of group
                     * LABEL_J: Translational current of group
    """
    def __init__(self,
                 atomgroups,
                 restypes=None,
                 output_prefix="",
                 labels=None,
                 nojump=False,
                 bpbc=True,
                 **kwargs):
        super().__init__(atomgroups, **kwargs)

        # Group types of each AG
        if restypes is not None:
            if len(restypes) != len(self.atomgroups):
                raise ValueError(
                    "Number of atomgroups and residue types not equal.")
        else:
            restypes = ["CM".format(i) for i in range(len(self.atomgroups))]
        self.restypes = restypes
<<<<<<< HEAD

=======
        
>>>>>>> 05a4352 (Added module for calculating spectra with free charges)
        # Add check for labels and residues
        # Names of each AG
        if labels is not None:
            if len(labels) != len(self.atomgroups):
                raise ValueError(
                    "Number of atomgroups and label names not equal.")
        else:
            labels = ["ag{}".format(i) for i in range(len(self.atomgroups))]
        self.labels = labels

        self.nojump = nojump
        self.bpbc = bpbc
        self.output_prefix = output_prefix

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output_prefix')
        parser.add_argument('-r', dest='restypes', nargs="+")
        parser.add_argument('-l', dest='labels', nargs="+")
        parser.add_argument('-nojump', dest='nojump')
        parser.add_argument('-nopbcrepair', dest='bpbc')

    def _prepare(self):
        self.volume = 0.0

        # Setup the results arrays
        self.results["dt"] = self._trajectory.dt * self.step
        self.results["time"] = np.round(
            self._trajectory.dt * np.arange(self.startframe, self.stopframe, self.step),
            decimals=4)

        for i, ag in enumerate(self.atomgroups):
            label = self.labels[i]
            self.results[label + "_MD"] = np.zeros((self.n_frames, 3))
            self.results[label + "_J"] = np.zeros((self.n_frames, 3))

            if self.nojump:
                self.results[label + "_MJ"] = np.zeros((self.n_frames, 3))

    def _single_frame(self):
        # Increment volume
        self.volume += self._ts.volume

        # Loop over each AG, gets a vector for P and J
        for i, ag in enumerate(self.atomgroups):

            if self.bpbc:
                # make broken molecules whole again!
                ag.unwrap(compound="molecules")

            # Determines which calculation method to use for each AG
            label = self.labels[i]
            restype = self.restypes[i]

            if restype == "SP":
                # Single particle
                # No COM dipole, only position/velocity current if charged
                MD = np.zeros(3)
                J = np.dot(ag.charges, ag.velocities)
                if self.nojump:
                    MJ = np.dot(ag.charges, ag.positions)

            elif restype == "NM":
                # Neutral molecule
                # No dipole/velocity current, only rotational dipole
                MD = np.dot(ag.charges, ag.positions)
                J = np.zeros(3)
                if self.nojump:
                    MJ = np.zeros(3)

            else:
                # Generic calculation for charged molecules
                # Vectorized calculation for each residue in the group
                idx = np.argwhere(ag.residues.resids[:,
                                                     np.newaxis] == ag.resids)
                idx = np.asarray(
                    np.split(
                        idx[:, 1],
                        np.cumsum(np.unique(idx[:, 0],
                                            return_counts=True)[1])[:-1]))

                pos = ag.positions[idx]
                vel = ag.velocities[idx]
                ms = ag.masses[idx]
                qs = ag.charges[idx]

                mtot = np.sum(ms, axis=1, keepdims=True)
                qtot = np.sum(qs, axis=1, keepdims=True)

                rcm = np.sum(ms[:, :, np.newaxis] * pos, axis=1) / mtot
                vcm = np.sum(ms[:, :, np.newaxis] * vel, axis=1) / mtot

                MD = np.sum(np.sum(qs[:, :, np.newaxis] *
                                   (pos - rcm[:, np.newaxis, :]),
                                   axis=1),
                            axis=0)
                J = np.sum(qtot * vcm, axis=0)
                if self.nojump:
                    MJ = np.sum(qtot * rcm, axis=0)

            self.results[label + "_MD"][self._frame_index, :] = MD
            self.results[label + "_J"][self._frame_index, :] = J
            if self.nojump:
                self.results[label + "_MJ"][self._frame_index, :] = MJ

    def _conclude(self):
        self.volume = self.volume / (self._frame_index + 1)
        self.results["volume"] = self.volume
        self.results["nframes"] = self._frame_index + 1

    def _save_results(self):
        # Save volumne
        savetxt(self.output_prefix + "volume.dat", [self.results["volume"]],
                header="Avg. Volume")

        for label in self.labels:
            if self.nojump:
                vals = (self.results["time"], self.results[label + "_MD"],
                        self.results[label + "_MJ"],
                        self.results[label + "_J"])
                header = "time, MDx, MDy, MDz, MJx, MJy, MJz, Jx, Jy, Jz"
            else:
                vals = (self.results["time"], self.results[label + "_MD"],
                        self.results[label + "_J"])
                header = "time, MDx, MDy, MDz, Jx, Jy, Jz"

            savetxt(self.output_prefix + "diptrj_" + label + ".dat",
                    np.column_stack(vals),
                    header=header)
