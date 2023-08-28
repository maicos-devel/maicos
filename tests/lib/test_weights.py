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
from maicos.lib.util import unit_vectors_planar


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import SPCE_GRO, SPCE_ITP, WATER_TPR, WATER_TRR  # noqa: E402


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


class Testdiporder_weights:
    """Test the dipolar weights "base" function.

    In details tests are designed to check if the scalar product in performed
    correctly.
    """

    @pytest.fixture
    def atomgroup(self):
        """Atomgroup containing a single water molecule poiting in z-direction."""
        return mda.Universe(SPCE_ITP, SPCE_GRO).atoms

    @pytest.mark.parametrize("pdim, P0", [(0, 0), (1, 0), (2, 0.491608)])
    def test_P0(self, atomgroup, pdim, P0):
        """Test calculation of the projection of the dipole moment."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=pdim)

        diporder_weights = maicos.lib.weights.diporder_weights(
            atomgroup=atomgroup,
            grouping="fragments",
            order_parameter="P0",
            get_unit_vectors=get_unit_vectors,
        )

        assert_allclose(diporder_weights, np.array([P0]))

    @pytest.mark.parametrize("pdim, cos_theta", [(0, 0), (1, 0), (2, 1)])
    def test_cos_theta(self, atomgroup, pdim, cos_theta):
        """Test calculation of the cos of the dipole moment and a unit vector."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=pdim)

        diporder_weights = maicos.lib.weights.diporder_weights(
            atomgroup=atomgroup,
            grouping="fragments",
            order_parameter="cos_theta",
            get_unit_vectors=get_unit_vectors,
        )

        assert_allclose(diporder_weights, np.array([cos_theta]))

    def test_cos_2_theta(self, atomgroup):
        """Test that cos_2_theta is the squared of cos_theta."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=0)

        kwargs_weights = {
            "atomgroup": atomgroup,
            "grouping": "fragments",
            "get_unit_vectors": get_unit_vectors,
        }

        assert_equal(
            maicos.lib.weights.diporder_weights(
                order_parameter="cos_theta", **kwargs_weights
            )
            ** 2,
            maicos.lib.weights.diporder_weights(
                order_parameter="cos_2_theta", **kwargs_weights
            ),
        )

    def test_wrong_unit_vector_shape(self, atomgroup):
        """Test raise for a wrong shape of provided unit vector."""

        def get_unit_vectors(atomgroup, grouping):
            return np.zeros([10, 4])

        match = (
            r"Returned unit vectors have shape \(10, 4\). But only shape \(3,\) or "
            r"\(1, 3\) is allowed."
        )

        with pytest.raises(ValueError, match=match):
            maicos.lib.weights.diporder_weights(
                atomgroup=atomgroup,
                grouping="fragments",
                order_parameter="cos_theta",
                get_unit_vectors=get_unit_vectors,
            )

    def test_wrong_grouping(self, atomgroup):
        """Test error raise for wrong grouping."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=0)

        with pytest.raises(ValueError, match="'foo' not supported."):
            maicos.lib.weights.diporder_weights(
                atomgroup=atomgroup,
                grouping="fragments",
                order_parameter="foo",
                get_unit_vectors=get_unit_vectors,
            )
