#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VelocityCylinder class."""

import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from maicos import VelocityCylinder

sys.path.append(str(Path(__file__).parents[1]))
from util import circle_of_water_molecules  # noqa: E402


class TestVelocityCylinder:
    """Tests for the VelocityCylinder class."""

    @pytest.fixture
    def vel_array(self):
        """Set velocity array for test_vel_cylinder."""
        array = []
        array.append([1, np.nan, np.nan, np.nan, np.nan])  # rad=0
        array.append([np.nan, 1, np.nan, np.nan, np.nan])  # rad=3
        array.append([np.nan, np.nan, 1, np.nan, np.nan])  # rad=5
        array.append([np.nan, np.nan, np.nan, 1, np.nan])  # rad=7.5
        return array

    @pytest.mark.parametrize(
        ("radius", "data_index"), [(0, 0), (3, 1), (5, 2), (7.5, 3)]
    )
    def test_vel_cylinder(self, vel_array, radius, data_index):
        """Test VelocityCylinder module.

        Create a universe with 10 water molecules along a circle (in the (x,y) plan) of
        radius equal to radius, with an imposed velocity of 1 along z.

        Call VelocityCylinder module to measure, using a bin width of 2, and a grouping
        per molecule. data_index checks that we check all bins.`

        Do the same test for measuring the flux, i.e., normalization is done per volume
        instead per molecule.
        """
        ag_v, bin_volume = circle_of_water_molecules(
            myvel=np.array([0, 0, 1]), bin_width=2, radius=radius
        )
        vel = VelocityCylinder(
            ag_v, vdim=2, bin_width=2, grouping="molecules", refgroup=ag_v
        ).run()
        assert_allclose(vel.results.profile, vel_array[data_index])

        # Test flux, 10 Molecules
        flux = VelocityCylinder(
            ag_v, vdim=2, bin_width=2, grouping="molecules", flux=True
        ).run()
        assert_allclose(
            flux.results.profile.T,
            np.nan_to_num(vel_array[data_index] / (bin_volume / 10)),
        )
