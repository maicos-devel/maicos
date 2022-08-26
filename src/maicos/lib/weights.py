#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Weighting functions."""

import numpy as np
from scipy import constants


def density_weights(atomgroup, grouping, dim, dens):
    """Calculate the weights for the histogram.

    Supported values are `mass`, `number` or `charge`.
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


def temperature_weights(ag, grouping, dim):
    """Calculate contribution of each atom to the temperature."""
    # ((1 amu * Å^2) / (ps^2)) / Boltzmann constant
    prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
    return (ag.velocities ** 2).sum(axis=1) * ag.atoms.masses / 2 * prefac


def velocity_weights(atomgroup, grouping, dim, vdim, flux):
    """Calculate the weights for the velocity histogram."""
    atom_vels = atomgroup.velocities[:, vdim]

    if grouping == "atoms":
        vels = atom_vels
    else:
        mass_vels = atomgroup.atoms.accumulate(
            atom_vels * atomgroup.atoms.masses, compound=grouping)
        group_mass = atomgroup.atoms.accumulate(
            atomgroup.atoms.masses, compound=grouping)
        vels = mass_vels / group_mass

    # either normalised by the number of compound (to get the velocity)
    # or do not normalise to get the flux (velocity x number of compound)
    if not flux:
        vels /= len(vels)
    return vels
