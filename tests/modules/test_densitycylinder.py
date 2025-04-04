#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the density cylinder module."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import DensityCylinder

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_GRO_NPT, WATER_TPR_NPT  # noqa: E402
from util import circle_of_water_molecules  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms


class TestDensityCylinder(ReferenceAtomGroups):
    """Tests for the DensityCylinder class."""

    @pytest.mark.parametrize(
        ("dens_type", "mean"),
        [("mass", 0.561), ("number", 0.095), ("charge", 0.000609)],
    )
    def test_actual_simulation(self, ag_single_frame, dens_type, mean):
        """Test DensityCylinder from a single frame.

        Import a single frame universe and measure the density.
        """
        dens = DensityCylinder(ag_single_frame, dens=dens_type).run()
        assert_allclose(dens.results.profile.mean(), mean, atol=1e-4, rtol=1e-2)

    def test_regularly_spaced_molecule(self):
        """Test VelocityCylinder module using regularly spaced molecules.

        Create a universe with 10 water molecules along a circle of radius equal to 5
        Angstroms.

        Call DensityCylinder module to measure the density, using a bin width of 2, and
        a grouping per molecule.
        """
        n_molecule = 10
        bin_width = 2
        ag, volume_slices = circle_of_water_molecules(
            n_molecules=n_molecule, radius=5, bin_width=bin_width
        )
        dens = DensityCylinder(
            ag, bin_width=bin_width, dens="number", refgroup=ag
        ).run()
        assert_allclose(
            dens.results.profile,
            np.array([0, 0, 3 * n_molecule, 0, 0]) / volume_slices,
        )
