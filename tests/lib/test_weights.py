#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import maicos.lib.weights


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_TPR, WATER_TRR  # noqa: E402


def test_density_weights_mass():
    """Test mass weights."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, "atoms", "mass")
    # Convert back to atomic units
    assert_allclose(weights, u.atoms.masses)


@pytest.mark.parametrize("compound", ["residues", "segments", "molecules", "fragments"])
def test_density_weights_mass_grouping(compound):
    """Test mass weights with grouping."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, compound, "mass")
    assert_allclose(weights, u.atoms.total_mass(compound=compound))


def test_density_weights_charge():
    """Test charge weights."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, "atoms", "charge")
    assert_equal(weights, u.atoms.charges)


@pytest.mark.parametrize("compound", ["residues", "segments", "molecules", "fragments"])
def test_density_weights_charge_grouping(compound):
    """Test charge weights with grouping."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, compound, "charge")
    assert_equal(weights, u.atoms.total_charge(compound=compound))


@pytest.mark.parametrize("compound", ["residues", "segments", "fragments"])
def test_density_weights_number(compound):
    """Test number weights for grouping."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, compound, "number")
    assert_equal(weights, np.ones(getattr(u.atoms, f"n_{compound}")))


def test_density_weights_number_molecules():
    """Test number weights for grouping with respect to molecules."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    weights = maicos.lib.weights.density_weights(u.atoms, "molecules", "number")
    assert_equal(weights, np.ones(len(np.unique(u.atoms.molnums))))


def test_density_weights_error():
    """Test error raise for non existing weight."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    with pytest.raises(ValueError, match="not supported"):
        maicos.lib.weights.density_weights(u.atoms, "atoms", "foo")


@pytest.mark.parametrize("grouping", ("residues", "segments", "molecules", "fragments"))
def test_tempetaure_weights_grouping(grouping):
    """Test when grouping != atoms."""
    u = mda.Universe(WATER_TPR, WATER_TRR)
    with pytest.raises(NotImplementedError):
        maicos.lib.weights.temperature_weights(u.atoms, grouping)
