#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper and utilities functions for testing."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import sympy as sp

sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402


def error_prop(f, variables, errors):
    """Analytic error propagation.

    This function takes a sympy expression, its variables, and
    the corresponding error values as inputs and returns the analytic
    error propagation as a regular python function of the variables.
    """
    assert len(variables) == len(errors)
    error_propagation = 0
    for i, var in enumerate(variables):
        error_propagation += (np.abs(f.diff(var)) * errors[i]) ** 2

    return sp.lambdify([*variables], sp.sqrt(error_propagation))


# Useful functions for creating test systems


def line_of_water_molecules(
    n_molecules: int = 1,
    distance: float = 10.0,
    angle_deg: float | list[float] = 0.0,
    axis_rotation: tuple[float, float, float] = (0.0, 1.0, 0.0),
    myvel: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Create an MDAnalysis universe with regularly spaced molecules.

    The molecules are placed along a line `distance` apart, have an orientation
    controlled by `angle_deg` and `axis_rotation`. All the molecules have the same
    velocities `myvel`.

    Parameters
    ----------
    n_molecules : int
        number of molecules to create.
    distance : float
        distance between the molecules.
    angle_deg : float, List[float]
        angle by which the molecules will be rotated. If `angle_deg` is float all
        molecules will be rotated by the same angle. If List[float] each molecule will
        be rotated by a different angle.
    axis_rotation : tuple[float, float, float]
        axis of rotation.
    myvel : tuple[float, float, float]
        velocity of the molecules.
    """
    # import molecule topology
    fluid = []
    for _n in range(n_molecules):
        fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp"))

    # define evenly spaced positions along a line
    positions = []
    for _n in range(n_molecules):
        positions.append((0, 0, distance * (1 * _n)))

    # set the orientation of the molecules
    rotations = []
    if type(angle_deg) is float:
        angle_deg_list = n_molecules * [angle_deg]
    elif type(angle_deg) is list:
        angle_deg_list = angle_deg
    else:
        raise ValueError(
            f"type of angle_deg ({type(angle_deg)}) is neither a list or float!"
        )

    if len(angle_deg_list) != n_molecules:
        raise ValueError(
            f"Length of {len(angle_deg_list)} is not the same a n_molecules "
            f"{n_molecules}."
        )

    for _n, angle in zip(range(n_molecules), angle_deg_list, strict=True):
        rotations.append([angle, axis_rotation])

    # multiply molecules and apply translation and rotations
    for molecule, rotation, position in zip(fluid, rotations, positions, strict=True):
        molecule.atoms.rotateby(rotation[0], rotation[1])
        molecule.atoms.translate(position)

    # merges the molecules into a universe
    u = mda.Merge(*[molecule.atoms for molecule in fluid])

    # set the universe's dimension
    _dimensions = fluid[0].dimensions
    _dimensions[2] = distance * n_molecules
    u.dimensions = _dimensions

    # set residue ids
    u.residues.molnums = list(range(1, n_molecules + 1))

    # give velocities to the molecules
    u.trajectory.ts.has_velocities = True
    u.atoms.velocities += np.array(myvel)
    return u.select_atoms("name OW HW1 HW2")


def circle_of_water_molecules(
    n_molecules=10,
    angle_deg=0,
    axis_rotation=(0, 1, 0),
    myvel=(0, 0, 0),
    radius=5,
    bin_width=1,
):
    """Create a ``MDAnalysis.Universe`` with regularly spaced molecules.

    Molecules are placed on a circle of radius `radius` around the box center. The box
    dimensions are set to 20x20x20. The radius must be smaller than 10!
    """
    if radius > 10:
        raise ValueError("radius has to be smaller than 10")

    # import molecule topology
    fluid = []
    for _n in range(n_molecules):
        fluid.append(mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp"))

    # define evenly spaced positions along a circle
    positions = []
    for _n in range(0, n_molecules):
        x = np.cos(2 * np.pi / n_molecules * _n) * radius
        y = np.sin(2 * np.pi / n_molecules * _n) * radius
        z = 0
        positions.append(10 + np.array([x, y, z]))

    # set the orientation of the molecules
    rotations = []
    for _n in range(n_molecules):
        rotations.append([angle_deg, axis_rotation])

    # multiply molecules and apply translation and rotations
    for molecule, rotation, position in zip(fluid, rotations, positions, strict=True):
        molecule.atoms.rotateby(rotation[0], rotation[1])
        molecule.atoms.translate(position)

    # merges the molecules into a universe
    u = mda.Merge(*[molecule.atoms for molecule in fluid])

    # set the universe's dimension
    u.dimensions = (20, 20, 20, 90, 90, 90)

    # set residue ids
    u.residues.molnums = list(range(1, n_molecules + 1))

    # give velocities to the molecules
    u.trajectory.ts.has_velocities = True
    u.atoms.velocities += np.array(myvel)

    # return the volume of each slice
    rmin = 0
    rmax = u.dimensions[0] / 2

    zmin = 0
    zmax = u.dimensions[2]
    L = zmax - zmin
    n_bins = np.int32((rmax - rmin) / bin_width)
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    bin_area = np.pi * np.diff(bin_edges**2)
    volume = bin_area * L
    return u.select_atoms("name OW HW1 HW2"), volume
