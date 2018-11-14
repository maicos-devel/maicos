#!/usr/bin/env python
# coding: utf-8

# Mandatory imports
from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np

from .base import AnalysisBase

class kinetic_energy(AnalysisBase):
    """Calculates the timeseries for the molecular center
       translational and rotational kinetic energy."""

    def __init__(self, atomgroup, output="output", refpoint="COM", **kwargs):
        # Inherit all classes from AnalysisBase
        super(kinetic_energy, self).__init__(atomgroup.universe.trajectory,
                                             **kwargs)

        self.atomgroup = atomgroup
        self.output = output
        self.refpoint = refpoint

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument('-o', dest='output', type=str, default='ke', 
                            help='Prefix for output filenames')
        parser.add_argument('-r', dest='refpoint', type=str, default='COM', choices=["COM", "COC", "OXY"],
                            help='reference point for molecular center: center of' +
                                 ' mass (COM), center of charge (COC), or oxygen position (OXY)' +
                                 'Note: The oxygen position only works for systems of pure water')
                                
    def _prepare(self):
        """Set things up before the analysis loop begins"""
        self.atomsPerMolecule = []
        self.seg_masses = []
        self.seg_abscharges = []
        for j, seg in enumerate(self.atomgroup.segments):
            self.atomsPerMolecule.append(seg.atoms.n_atoms // seg.atoms.n_residues)
            self.seg_masses.append(seg.residues.masses)
            self.seg_abscharges.append(sum(np.abs(seg.atoms.charges)[
                i::self.atomsPerMolecule[j]] for i in range(self.atomsPerMolecule[j])))

        if self.refpoint == "OXY":
            self.oxy = self.atomgroup.select_atoms("name OW*")
        
        # Total kinetic energy
        self.E_kin = np.zeros(self.n_frames)

        # Molecular center energy
        self.E_center = np.zeros(self.n_frames)

    def _single_frame(self):
        self.E_kin[self._frame_index] = np.dot(
            self.atomgroup.masses, np.linalg.norm(self.atomgroup.velocities, axis=1)**2)

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
                self.seg_masses[j], np.linalg.norm(v, axis=1)**2)

    def _calculate_results(self):
        self.results["t"] = self._trajectory.dt * \
            np.arange(self.start, self.stop, self.step)
        self.results["trans"] = self.E_center / 2
        self.results["rot"] = (self.E_kin - self.E_center) / 2

    def _save_results(self):
        np.savetxt("{}.dat".format(self.output),
                   np.vstack(
                       [self.results["t"], self.results["trans"], self.results["rot"]]).T,
                   fmt='%.8e', header="t / ps \t E_kin^trans \t E_kin^rot")
