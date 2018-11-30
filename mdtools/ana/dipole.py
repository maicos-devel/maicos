#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import numpy as np

from ..utils import repairMolecules
from .base import AnalysisBase


class dipole_angle(AnalysisBase):
    """Calculates the timeseries of the dipole moment wit an axis."""

    def __init__(self, atomgroup, output="output", dim=2, sel='all', **kwargs):
        # Inherit all classes from AnalysisBase
        super(dipole_angle, self).__init__(atomgroup.universe.trajectory,
                                           **kwargs)

        self.atomgroup = atomgroup
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
            '-sel',
            dest='sel',
            type=str,
            help='atom group selection',
            default='resname SOL')
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

        self.sol = self.atomgroup.select_atoms(self.sel)

        self.n_residues = self.sol.residues.n_residues
        self.atomsPerMolecule = self.sol.n_atoms // self.n_residues

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

        dt = self.sol.universe.trajectory.dt * self.step

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self):

        # make broken molecules whole again!
        repairMolecules(self.atomgroup)

        chargepos = self.sol.atoms.positions * \
            self.sol.atoms.charges[:, np.newaxis]
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
            np.arange(self.start, self.stop, self.step)

        self.results["cos_theta_i"] = self.cos_theta_i[:self._index]
        self.results["cos_theta_ii"] = self.cos_theta_ii[:self._index]
        self.results["cos_theta_ij"] = self.cos_theta_ij[:self._index]

    def _save_results(self):

        np.savetxt(
            "{}.dat".format(self.output),
            np.vstack([
                self.results["t"], self.results["cos_theta_i"],
                self.results["cos_theta_ii"], self.results["cos_theta_ij"]
            ]).T,
            header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
            fmt='%.5e')
