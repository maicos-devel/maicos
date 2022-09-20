#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weight functions used for spatial binned analysis modules."""

import numpy as np
from scipy import constants

from .util import get_compound


def density_weights(atomgroup, grouping, dens):
    """Weights for density calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : str, {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to
    dens : str, {'mass', 'number', 'charge'}
        type of density weight

    Returns
    -------
    numpy.ndarray
        1D array of calculated weights. The length depends on the grouping.
    """
    if dens == "number":
        # There exist no properrty like n_molecules
        if grouping == "molecules":
            numbers = len(np.unique(atomgroup.molnums))
        else:
            numbers = getattr(atomgroup, f"n_{grouping}")
        return np.ones(numbers)
    elif dens == "mass":
        if grouping == "atoms":
            masses = atomgroup.masses
        else:
            masses = atomgroup.total_mass(compound=grouping)
        return masses
    elif dens == "charge":
        if grouping == "atoms":
            return atomgroup.charges
        else:
            return atomgroup.total_charge(compound=grouping)
    else:
        raise ValueError(f"`{dens}` not supported. "
                         "Use `mass`, `number` or `charge`.")


def temperature_weights(ag, grouping):
    """Weights for temperature calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : str, {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to

    Returns
    -------
    numpy.ndarray
        1D array of calculated weights. The length depends on the grouping.

    Raises
    ------
    NotImplementedError
        Currently only works for `grouping='atoms'`
    """
    if grouping != "atoms":
        raise NotImplementedError(f"Temperature calculations of '{grouping}'"
                                  "is not supported. Use 'atoms' "
                                  "instead.'")

    # ((1 amu * Å^2) / (ps^2)) / Boltzmann constant
    prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
    return (ag.velocities ** 2).sum(axis=1) * ag.atoms.masses / 2 * prefac


def diporder_weights(atomgroup, grouping, dim, order_parameter):
    """Weights for diporder calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : str, {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to
    dim : {0, 1, 2}
        direction of the projection
    order_parameter : str, {'P0', 'cos_theta', 'cos_2_theta'}
        type of weight to be calculated

    Returns
    -------
    numpy.ndarray
        1D array of calculated weights. The length depends on the grouping.
    """
    if grouping == "atoms":
        raise ValueError("Atoms do not have an orientation.")

    chargepos = atomgroup.positions * atomgroup.charges[:, np.newaxis]
    dipoles = atomgroup.accumulate(chargepos,
                                   compound=get_compound(atomgroup))

    # unit normal vector
    unit = np.zeros(3)
    unit[dim] += 1

    if order_parameter == "P0":
        weights = np.dot(dipoles, unit)
    elif order_parameter in ["cos_theta", "cos_2_theta"]:
        weights = np.dot(dipoles
                         / np.linalg.norm(dipoles, axis=1)[:, np.newaxis],
                         unit)
        if order_parameter == "cos_2_theta":
            weights *= weights
    else:
        raise ValueError(f"'{order_parameter}' not supported. "
                         "Use 'P0', 'cos_theta' or 'cos_2_theta'.")

    return weights


def velocity_weights(atomgroup, grouping, vdim, flux):
    """Weights for velocity calculations.

    The function either normalises by the number of compounds
    (to get the velocity) or does not normalise to get the flux
    (flux = velocity x number of compounds).

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : str, {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to
    vdim : int, {0, 1, 2}
        direction of the velocity taken for the weights
    flux : bool
        convert velocities into a flux

    Returns
    -------
    numpy.ndarray
        1D array of calculated weights. The length depends on the grouping.
    """
    atom_vels = atomgroup.velocities[:, vdim]

    if grouping == "atoms":
        vels = atom_vels
    else:
        mass_vels = atomgroup.atoms.accumulate(
            atom_vels * atomgroup.atoms.masses, compound=grouping)
        group_mass = atomgroup.atoms.accumulate(
            atomgroup.atoms.masses, compound=grouping)
        vels = mass_vels / group_mass

    if not flux:
        vels /= len(vels)
    return vels
