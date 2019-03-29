#!/usr/bin/env python3
# coding: utf-8

import numpy as np

from ..utils import repairMolecules, savetxt
from .base import SingleGroupAnalysisBase


class dipole_angle(SingleGroupAnalysisBase):
    """Calculates the timeseries of the dipole moment with respect to an axis."""

    def __init__(self, atomgroup, output="output", dim=2, sel='all', **kwargs):
        super(dipole_angle, self).__init__(atomgroup, **kwargs)
        self.output = output
        self.dim = dim
        self.sel = sel

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-d',
            dest='dim',
            type=int,
            default=2,
            help='direction normal to the surface (x,y,z=0,1,2, default: z)')
        parser.add_argument(
            '-dout',
            dest='outfreq',
            type=float,
            default='10000',
            help=
            'Default number of frames after which output files are refreshed (10000)'
        )
        parser.add_argument(
            '-o',
            dest='output',
            type=str,
            default='dipangle',
            help='Prefix for output filenames')

    def _prepare(self):
        self.n_residues = self.atomgroup.residues.n_residues
        self.atomsPerMolecule = self.atomgroup.n_atoms // self.n_residues

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self):

        # make broken molecules whole again!
        repairMolecules(self.atomgroup)

        chargepos = self.atomgroup.positions * \
            self.atomgroup.charges[:, np.newaxis]
        dipoles = sum(chargepos[i::self.atomsPerMolecule]
                      for i in range(self.atomsPerMolecule))

        cos_theta = np.dot(dipoles, self.unit) / \
            np.linalg.norm(dipoles, axis=1)
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        self.cos_theta_i[self._frame_index] = cos_theta.mean()
        self.cos_theta_ii[self._frame_index] = trace / self.n_residues
        self.cos_theta_ij[self._frame_index] = (matrix.sum() - trace)
        self.cos_theta_ij[self._frame_index] /= (
            self.n_residues**2 - self.n_residues)

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

        savetxt(
            "{}.dat".format(self.output),
            np.vstack([
                self.results["t"], self.results["cos_theta_i"],
                self.results["cos_theta_ii"], self.results["cos_theta_ij"]
            ]).T,
            header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
            fmt='%.5e')


class kinetic_energy(SingleGroupAnalysisBase):
    """Calculates the timeseries for the molecular center
       translational and rotational kinetic energy (kJ/mole)."""

    def __init__(self, atomgroup, output="output", refpoint="COM", **kwargs):
        super(kinetic_energy, self).__init__(atomgroup, **kwargs)
        self.output = output
        self.refpoint = refpoint

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument(
            '-o',
            dest='output',
            type=str,
            default='ke',
            help='Prefix for output filenames')
        parser.add_argument(
            '-r',
            dest='refpoint',
            type=str,
            default='COM',
            choices=["COM", "COC", "OXY"],
            help='reference point for molecular center: center of' +
            ' mass (COM), center of charge (COC), or oxygen position (OXY)' +
            'Note: The oxygen position only works for systems of pure water')

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        self.atomsPerMolecule = []
        self.seg_masses = []
        self.seg_abscharges = []
        for j, seg in enumerate(self.atomgroup.segments):
            self.atomsPerMolecule.append(
                seg.atoms.n_atoms // seg.atoms.n_residues)
            self.seg_masses.append(seg.residues.masses)
            self.seg_abscharges.append(
                sum(
                    np.abs(seg.atoms.charges)[i::self.atomsPerMolecule[j]]
                    for i in range(self.atomsPerMolecule[j])))

        if self.refpoint == "OXY":
            self.oxy = self.atomgroup.select_atoms("name OW*")

        # Total kinetic energy
        self.E_kin = np.zeros(self.n_frames)

        # Molecular center energy
        self.E_center = np.zeros(self.n_frames)

    def _single_frame(self):
        self.E_kin[self._frame_index] = np.dot(
            self.atomgroup.masses,
            np.linalg.norm(self.atomgroup.velocities, axis=1)**2)

        for j, seg in enumerate(self.atomgroup.segments):
            if self.refpoint == "COM":
                massvel = seg.atoms.velocities * \
                    seg.atoms.masses[:, np.newaxis]
                v = sum(massvel[i::self.atomsPerMolecule[j]]
                        for i in range(self.atomsPerMolecule[j]))
                v /= self.seg_masses[j][:, np.newaxis]

            elif self.refpoint == "COC":
                abschargevel = seg.atoms.velocities * \
                    np.abs(seg.atoms.charges)[:, np.newaxis]
                v = sum(abschargevel[i::self.atomsPerMolecule[j]]
                        for i in range(self.atomsPerMolecule[j]))
                v /= self.seg_abscharges[j][:, np.newaxis]

            elif self.refpoint == "OXY":
                v = self.oxy.velocities

            self.E_center[self._frame_index] += np.dot(
                self.seg_masses[j],
                np.linalg.norm(v, axis=1)**2)

    def _calculate_results(self):
        self.results["t"] = self._trajectory.dt * \
            np.arange(self.startframe, self.stopframe, self.step)
        self.results["trans"] = self.E_center / 2 / 100
        self.results["rot"] = (self.E_kin - self.E_center) / 2 / 100

    def _save_results(self):
        savetxt(
            "{}.dat".format(self.output),
            np.vstack(
                [self.results["t"], self.results["trans"],
                 self.results["rot"]]).T,
            fmt='%.8e',
            header="t / ps \t E_kin^trans / kJ/mole \t E_kin^rot / kJ/mole")
