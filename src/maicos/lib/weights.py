#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weight functions used for spatial binned analysis modules."""

import MDAnalysis as mda
import numpy as np
from scipy import constants

from .util import Unit_vector, render_docs


def density_weights(atomgroup: mda.AtomGroup, grouping: str, dens: str) -> np.ndarray:
    """Weights for density calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to.
    dens : {'mass', 'number', 'charge'}
        type of density weight

    Returns
    -------
    numpy.ndarray
        1D array of calculated weights. The length depends on the grouping.

    Raises
    ------
    ValueError
        if grouping or dens parameter is not supported.
    """
    if grouping not in ["atoms", "residues", "segments", "molecules", "fragments"]:
        raise ValueError(
            f"{grouping!r} grouping is not supported. "
            "Use `atoms`, `residues`, `segments`, `molecules` or `fragments`."
        )

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
        raise ValueError(
            f"{dens!r} density type is not supported. "
            "Use `mass`, `number` or `charge`."
        )


def temperature_weights(atomgroup: mda.AtomGroup, grouping: str) -> np.ndarray:
    """Weights for temperature calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
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
        raise NotImplementedError(
            f"Temperature calculations of {grouping!r} is not supported. Use 'atoms' "
            "instead.'"
        )

    # ((1 u * Å^2) / (ps^2)) / Boltzmann constant
    prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
    return (atomgroup.velocities**2).sum(axis=1) * atomgroup.atoms.masses / 2 * prefac


@render_docs
def diporder_weights(
    atomgroup: mda.AtomGroup,
    grouping: str,
    order_parameter: str,
    get_unit_vectors: Unit_vector,
) -> np.ndarray:
    """Weights for general diporder calculations.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : {'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to
    ${ORDER_PARAMETER_PARAMETER}
    get_unit_vectors : Callable
        Callable that returns unit vectors on which the projection is performed.
        Returned unit_vectors can either be of shape (3,) or of shape (n, 3). For a
        shape of (3,) the same unit vector is used for all calculations.
    """
    dipoles = atomgroup.dipole_vector(compound=grouping)

    unit_vectors = get_unit_vectors(atomgroup=atomgroup, grouping=grouping)

    # Extend unit_vectors to be of the same length as of dipoles
    if unit_vectors.shape == (3,):
        np.tile(unit_vectors, len(dipoles)).reshape(len(dipoles), 3)
    elif not unit_vectors.shape == (len(dipoles), 3):
        raise ValueError(
            f"Returned unit vectors have shape {unit_vectors.shape}. But only shape "
            f"(3,) or {(len(dipoles), 3)} is allowed."
        )

    if order_parameter == "P0":
        weights = np.sum(dipoles * unit_vectors, axis=1)
    elif order_parameter in ["cos_theta", "cos_2_theta"]:
        weights = np.sum(
            dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis] * unit_vectors,
            axis=1,
        )
        if order_parameter == "cos_2_theta":
            weights *= weights
    else:
        raise ValueError(
            f"{order_parameter!r} not supported. "
            "Use 'P0', 'cos_theta' or 'cos_2_theta'."
        )

    return weights


def velocity_weights(atomgroup: mda.AtomGroup, grouping: str, vdim: int) -> np.ndarray:
    """Weights for velocity calculations.

    The function normalises by the number of compounds.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        atomgroup taken for weight calculation
    grouping : {'atoms', 'residues', 'segments', 'molecules', 'fragments'}
        constituent to group weights with respect to
    vdim : {0, 1, 2}
        direction of the velocity taken for the weights

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
            atom_vels * atomgroup.atoms.masses, compound=grouping
        )
        group_mass = atomgroup.atoms.accumulate(
            atomgroup.atoms.masses, compound=grouping
        )
        vels = mass_vels / group_mass

    return vels
