#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from ..utils import savetxt
from .base import SingleGroupAnalysisBase


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
                 output="dipangle",
                 outfreq=10000,
                 dim=2,
                 **kwargs):
        super(dipole_angle, self).__init__(atomgroup, **kwargs)
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
        dipoles = self.atomgroup.accumulate(chargepos, compound="molecules")

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

        savetxt("{}.dat".format(self.output),
                np.vstack([
                    self.results["t"], self.results["cos_theta_i"],
                    self.results["cos_theta_ii"], self.results["cos_theta_ij"]
                ]).T,
                header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
                fmt='%.5e')


class kinetic_energy(SingleGroupAnalysisBase):
    """Calculates the timeseries for the molecular center
       translational and rotational kinetic energy (kJ/mole).

       :param output (str): Prefix for output filenames.
       :param refpoint (str): reference point for molecular center: center of
                              mass (COM), center of charge (COC), or oxygen position (OXY)
                              Note: The oxygen position only works for systems of pure water

        :returns (dict): * t: time (ps)
                         * trans: translational kinetic energy (kJ/mole)
                         * rot: rotational kinetic energy (kJ/mole)
        """

    def __init__(self, atomgroup, output="ke", refpoint="COM", **kwargs):
        super(kinetic_energy, self).__init__(atomgroup, **kwargs)
        self.output = output
        self.refpoint = refpoint

    def _configure_parser(self, parser):
        parser.add_argument(
            '-o',
            dest='output',
        )
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
            self.atomgroup.atoms.masses, compound="molecules")
        self.abscharges = self.atomgroup.atoms.accumulate(np.abs(
            self.atomgroup.atoms.charges),
                                                          compound="molecules")
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
            v = self.atomgroup.accumulate(massvel, compound="molecules")
            v /= self.masses[:, np.newaxis]

        elif self.refpoint == "COC":
            abschargevel = self.atomgroup.velocities * \
                np.abs(self.atomgroup.charges)[:, np.newaxis]
            v = self.atomgroup.accumulate(abschargevel, compound="molecules")
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
        savetxt("{}.dat".format(self.output),
                np.vstack([
                    self.results["t"], self.results["trans"],
                    self.results["rot"]
                ]).T,
                fmt='%.8e',
                header="t / ps \t E_kin^trans / kJ/mole \t E_kin^rot / kJ/mole")
