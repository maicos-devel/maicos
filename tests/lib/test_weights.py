#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import maicos.lib.weights
from maicos.lib.tables import electron_count
from maicos.lib.util import unit_vectors_planar

sys.path.append(str(Path(__file__).parents[1]))
from data import (  # noqa: E402
    SALT_WATER_GRO,
    SPCE_GRO,
    SPCE_ITP,
    WATER_GRO_NPT,
    WATER_TPR_NPT,
    WATER_TRR_NPT,
)
from util import line_of_water_molecules  # noqa: E402


@pytest.fixture
def ag_spce():
    """Atomgroup containing a single water molecule poiting in z-direction."""
    return mda.Universe(SPCE_ITP, SPCE_GRO).atoms


@pytest.fixture
def ag_water_npt():
    """Water atomgroup in the NVT ensemble."""
    return mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT).atoms


def test_density_weights_mass(ag_water_npt):
    """Test mass weights."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, "atoms", "mass")
    # Convert back to atomic units
    assert_allclose(weights, ag_water_npt.masses)


@pytest.mark.parametrize("compound", ["residues", "segments", "molecules", "fragments"])
def test_density_weights_mass_grouping(ag_water_npt, compound):
    """Test mass weights with grouping."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, compound, "mass")
    assert_allclose(weights, ag_water_npt.total_mass(compound=compound))


def test_density_weights_charge(ag_water_npt):
    """Test charge weights."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, "atoms", "charge")
    assert_equal(weights, ag_water_npt.charges)


@pytest.mark.parametrize("compound", ["residues", "segments", "molecules", "fragments"])
def test_density_weights_charge_grouping(ag_water_npt, compound):
    """Test charge weights with grouping."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, compound, "charge")
    assert_equal(weights, ag_water_npt.total_charge(compound=compound))


@pytest.mark.parametrize("compound", ["residues", "segments", "fragments"])
def test_density_weights_number(ag_water_npt, compound):
    """Test number weights for grouping."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, compound, "number")
    assert_equal(weights, np.ones(getattr(ag_water_npt, f"n_{compound}")))


def test_density_weights_number_molecules(ag_water_npt):
    """Test number weights for grouping with respect to molecules."""
    weights = maicos.lib.weights.density_weights(ag_water_npt, "molecules", "number")
    assert_equal(weights, np.ones(len(np.unique(ag_water_npt.molnums))))


@pytest.mark.parametrize("compound", ["atoms", "residues", "segments"])
def test_density_weights_electron(compound):
    """Test electron weights with grouping."""
    u = mda.Universe(WATER_GRO_NPT)
    u.guess_TopologyAttrs(to_guess=["elements"])
    u.atoms.elements = np.array([el.title() for el in u.atoms.elements])

    d = {"H": 1, "O": 8}
    electrons = np.array([d[el] for el in u.atoms.elements], dtype=np.float64)
    weights = maicos.lib.weights.density_weights(u.atoms, compound, "electron")

    if compound == "atoms":
        assert_allclose(weights, electrons, rtol=1e-2)
    else:
        assert_allclose(
            actual=weights,
            desired=u.atoms.accumulate(electrons, compound=compound),
            rtol=1e-4,
        )


def test_density_weights_electron_title():
    """Test that the elements are converted to title case.

    SALT_WATER_GRO contains elements NA and CL in upper case.
    """
    u = mda.Universe(SALT_WATER_GRO)
    u.guess_TopologyAttrs(to_guess=["elements"])

    assert_allclose(
        maicos.lib.weights.density_weights(u.atoms, "atoms", "electron"),
        np.array([electron_count[el.title()] for el in u.atoms.elements]),
    )


@pytest.mark.parametrize(
    ("element", "n_electrons"),
    [("CH1", 7), ("CH2", 8), ("CH3", 9), ("NH1", 8), ("NH2", 9), ("NH3", 10)],
)
def test_density_weights_electron_united_atoms(element, n_electrons):
    """Test electron weights also for work for united atom force fields."""
    u = mda.Universe(SPCE_GRO)
    u.add_TopologyAttr("elements", u.atoms.n_atoms * [element])

    assert_allclose(
        maicos.lib.weights.density_weights(u.atoms, "atoms", "electron"),
        n_electrons * np.ones(u.atoms.n_atoms, dtype=np.float64),
        rtol=1e-3,
    )


def test_density_weights_electron_error():
    """Test error raise for non existing element."""
    u = mda.Universe(SPCE_GRO)
    u.add_TopologyAttr("elements", u.atoms.n_atoms * ["foo"])

    match = (
        "Element 'Foo' not found. Known elements are listed in the "
        "`maicos.lib.tables.elements` set."
    )
    with pytest.raises(KeyError, match=match):
        maicos.lib.weights.density_weights(u.atoms, "atoms", "electron")


def test_density_weights_error(ag_water_npt):
    """Test error raise for non existing weight."""
    with pytest.raises(ValueError, match="not supported"):
        maicos.lib.weights.density_weights(ag_water_npt, "atoms", "foo")


@pytest.mark.parametrize("grouping", ["residues", "segments", "molecules", "fragments"])
def test_tempetaure_weights_grouping(ag_water_npt, grouping):
    """Test when grouping != atoms."""
    with pytest.raises(NotImplementedError):
        maicos.lib.weights.temperature_weights(ag_water_npt, grouping)


def test_diporder_pair_weights_single(ag_spce):
    """Test that the weight of the same molecules is equal to one 1."""
    weights = maicos.lib.weights.diporder_pair_weights(
        ag_spce, ag_spce, compound="residues"
    )
    assert_allclose(weights, 1)


def test_diporder_pair_weights_line():
    """Test that the weight of the same molecules is equal to one 1."""
    ag = line_of_water_molecules(n_molecules=4, angle_deg=[0.0, 45.0, 90.0, 180.0])
    weights = maicos.lib.weights.diporder_pair_weights(ag, ag, compound="residues")

    weights_expected = np.array(
        [
            [1.00, 0.71, 0.00, -1.00],
            [0.71, 1.00, 0.71, -0.71],
            [0.00, 0.71, 1.00, 0.00],
            [-1.00, -0.71, 0.00, 1.00],
        ]
    )
    assert_equal(weights.round(2), weights_expected)


class Testdiporder_weights:
    """Test the dipolar weights "base" function.

    In details tests are designed to check if the scalar product in performed
    correctly.
    """

    @pytest.mark.parametrize(("pdim", "P0"), [(0, 0), (1, 0), (2, 0.491608)])
    def test_P0(self, ag_spce, pdim, P0):
        """Test calculation of the projection of the dipole moment."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=pdim)

        diporder_weights = maicos.lib.weights.diporder_weights(
            atomgroup=ag_spce,
            grouping="fragments",
            order_parameter="P0",
            get_unit_vectors=get_unit_vectors,
        )

        assert_allclose(diporder_weights, np.array([P0]))

    @pytest.mark.parametrize(("pdim", "cos_theta"), [(0, 0), (1, 0), (2, 1)])
    def test_cos_theta(self, ag_spce, pdim, cos_theta):
        """Test calculation of the cos of the dipole moment and a unit vector."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=pdim)

        diporder_weights = maicos.lib.weights.diporder_weights(
            atomgroup=ag_spce,
            grouping="fragments",
            order_parameter="cos_theta",
            get_unit_vectors=get_unit_vectors,
        )

        assert_allclose(diporder_weights, np.array([cos_theta]))

    def test_cos_2_theta(self, ag_spce):
        """Test that cos_2_theta is the squared of cos_theta."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=0)

        kwargs_weights = {
            "atomgroup": ag_spce,
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

    def test_wrong_unit_vector_shape(self, ag_spce):
        """Test raise for a wrong shape of provided unit vector."""

        def get_unit_vectors(atomgroup, grouping):  # noqa: ARG001
            return np.zeros([10, 4])

        match = (
            r"Returned unit vectors have shape \(10, 4\). But only shape \(3,\) or "
            r"\(1, 3\) is allowed."
        )

        with pytest.raises(ValueError, match=match):
            maicos.lib.weights.diporder_weights(
                atomgroup=ag_spce,
                grouping="fragments",
                order_parameter="cos_theta",
                get_unit_vectors=get_unit_vectors,
            )

    def test_wrong_order_parameter(self, ag_spce):
        """Test error raise for wrong order_parameter."""

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=0)

        with pytest.raises(ValueError, match="'foo' not supported."):
            maicos.lib.weights.diporder_weights(
                atomgroup=ag_spce,
                grouping="fragments",
                order_parameter="foo",
                get_unit_vectors=get_unit_vectors,
            )

    def test_atoms_grouping(self, ag_spce):
        """Test error raise if grouping="atoms".

        For atoms now dipoler moments are defined and we should test that a propper
        error is raised in this is option is provided.

        The error should is raised by MDAnalysis and we only test if this is the
        case.
        """

        def get_unit_vectors(atomgroup, grouping):
            return unit_vectors_planar(atomgroup, grouping, pdim=0)

        with pytest.raises(ValueError, match="Unrecognized compound definition: atoms"):
            maicos.lib.weights.diporder_weights(
                atomgroup=ag_spce,
                grouping="atoms",
                order_parameter="cos_thete",
                get_unit_vectors=get_unit_vectors,
            )
