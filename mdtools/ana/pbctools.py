#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import MDAnalysis
import numpy as np


def repairMolecules(selection):
    """Repairs molecules that are broken due to peridodic boundaries.
    To this end the center of mass is reset into the central box.
    CAVE: Only works with small (< half box) molecules."""

    # we repair each moleculetype individually for performance reasons
    for seg in selection.segments:
        atomsPerMolecule = seg.atoms.n_atoms // seg.atoms.n_residues

        # Make molecules whole, use first atom as reference
        distToFirst = np.empty((seg.atoms.positions.shape))
        for i in range(atomsPerMolecule):
            distToFirst[i::atomsPerMolecule] = seg.atoms.positions[i::atomsPerMolecule] - \
                seg.atoms.positions[0::atomsPerMolecule]
        seg.atoms.positions -= (np.abs(distToFirst) >
                                selection.dimensions[:3] / 2.) * selection.dimensions[:3] * np.sign(distToFirst)

        # Calculate the centers of the objects ( i.e. Molecules )
        masspos = (seg.atoms.positions * seg.atoms.masses[:, np.newaxis]).reshape(
            (seg.atoms.n_atoms // atomsPerMolecule, atomsPerMolecule, 3))
        # all molecules should have same mass
        centers = np.sum(masspos.T, axis=1).T / \
            seg.atoms.masses[:atomsPerMolecule].sum()

        # now shift them back into the primary simulation cell
        seg.atoms.positions += np.repeat(
            (centers % selection.dimensions[:3]) - centers, atomsPerMolecule, axis=0)
