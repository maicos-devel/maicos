#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weight functions used for spatial binned analysis modules."""

import MDAnalysis as mda
import numpy as np
from scipy import constants

from .math import compute_form_factor
from .util import Unit_vector, render_docs


@render_docs
def density_weights(atomgroup: mda.AtomGroup, grouping: str, dens: str) -> np.ndarray:
    """Weights for density calculations.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${GROUPING_PARAMETER}
    ${DENS_PARAMETER}

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
            f"'{grouping}' grouping is not supported. "
            "Use `atoms`, `residues`, `segments`, `molecules` or `fragments`."
        )

    if dens == "number":
        # There exist no properrty like n_molecules
        if grouping == "molecules":
            numbers = len(np.unique(atomgroup.molnums))
        else:
            numbers = getattr(atomgroup, f"n_{grouping}")
        weights = np.ones(numbers)
    elif dens == "mass":
        if grouping == "atoms":
            weights = atomgroup.masses
        else:
            weights = atomgroup.total_mass(compound=grouping)
    elif dens == "charge":
        if grouping == "atoms":
            weights = atomgroup.charges
        else:
            weights = atomgroup.total_charge(compound=grouping)
    elif dens == "electron":
        # Cromer-Mann parameters for q=0 is the number of electrons
        electrons = np.array(
            [compute_form_factor(q=0, element=el) for el in atomgroup.elements],
            dtype=np.float64,
        )
        if grouping == "atoms":
            weights = electrons
        else:
            weights = atomgroup.accumulate(electrons, compound=grouping)
    else:
        raise ValueError(
            f"'{dens}' density type is not supported. Use 'mass', 'number', 'charge' "
            "or 'electron'."
        )

    return weights


@render_docs
def temperature_weights(atomgroup: mda.AtomGroup, grouping: str) -> np.ndarray:
    """Weights for temperature calculations.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${GROUPING_PARAMETER}

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
            f"Temperature calculations of '{grouping}' is not supported. Use 'atoms' "
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
    ${ATOMGROUP_PARAMETER}
    ${GROUPING_PARAMETER}
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
    elif unit_vectors.shape != (len(dipoles), 3):
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
            f"'{order_parameter}' not supported. "
            "Use 'P0', 'cos_theta' or 'cos_2_theta'."
        )

    return weights


def diporder_pair_weights(
    g1: mda.AtomGroup, g2: mda.AtomGroup, compound: str
) -> np.ndarray:
    """Normalized dipole moments as weights for general diporder RDF calculations."""
    dipoles_1 = g1.dipole_vector(compound=compound)
    dipoles_2 = g2.dipole_vector(compound=compound)

    dipoles_1 /= np.linalg.norm(dipoles_1, axis=1)[:, np.newaxis]
    dipoles_2 /= np.linalg.norm(dipoles_2, axis=1)[:, np.newaxis]

    return dipoles_1 @ dipoles_2.T


@render_docs
def velocity_weights(atomgroup: mda.AtomGroup, grouping: str, vdim: int) -> np.ndarray:
    """Weights for velocity calculations.

    The function normalises by the number of compounds.

    Parameters
    ----------
    ${ATOMGROUP_PARAMETER}
    ${GROUPING_PARAMETER}
    ${VDIM_PARAMETER}

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
