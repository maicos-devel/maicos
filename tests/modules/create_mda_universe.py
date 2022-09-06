#!/usr/bin/env python3
"""Create a mda analysis universe with few molecules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

import MDAnalysis as mda
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402


def isolated_water_universe(n_molecules=1, angle_deg=0,
                            axis_rotation=(0, 1, 0), myvel=(0, 0, 0)):
    """Create a MDAnalysis universe with regularly spaced molecules."""
    # import molecule topology
    fluid = []
    for _n in range(n_molecules):
        fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO, topology_format='itp'))

    # set the orientation of the molecules
    rotations = []
    for _n in range(n_molecules):
        rotations.append([angle_deg, axis_rotation])

    # define evenly spaced positions for the molecules
    translations = []
    for _n in range(n_molecules):
        translations.append((0, 0, 10 * (1 * _n)))

    # multiply molecules and apply translation and rotations
    for molecule, rotation, translation in zip(fluid, rotations, translations):
        molecule.atoms.rotateby(rotation[0], rotation[1])
        molecule.atoms.translate(translation)

    # merges the molecules into a universe
    u = mda.Merge(*[molecule.atoms for molecule in fluid])

    # set the universe's dimension
    _dimensions = fluid[0].dimensions
    _dimensions[2] *= n_molecules
    u.dimensions = _dimensions

    # set residue ids
    u.residues.molnums = list(range(1, n_molecules + 1))

    # give velocities to the molecules
    u.trajectory.ts.has_velocities = True
    u.atoms.velocities += np.array(myvel)
    return u.select_atoms("name OW HW1 HW2")
