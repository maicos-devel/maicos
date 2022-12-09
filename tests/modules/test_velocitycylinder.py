#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the VelocityCylinder class."""
import os
import sys

import numpy as np
import pytest
from create_mda_universe import circle_of_water_molecules
from numpy.testing import assert_allclose

from maicos import VelocityCylinder


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TestVelocityCylinder(object):
    """Tests for the VelocityCylinder class."""

    @pytest.fixture()
    def vel_array(self):
        """Set velocity array for test_vel_cylinder."""
        array = []
        array.append([1, 0, 0, 0, 0])  # rad=0
        array.append([0, 1, 0, 0, 0])  # rad=2
        array.append([0, 0, 1, 0, 0])  # rad=5
        array.append([0, 0, 0, 1, 0])  # rad=7.5
        return array

    @pytest.mark.parametrize('rad', [(0, 0), (2, 1), (5, 2), (7.5, 3)])
    def test_vel_cylinder(self, vel_array, rad):
        """
        Test VelocityCylinder module.

        Create a universe with 10 water molecules
        along a circle (in the (x,y) plan) of radius equal to rad,
        with an imposed velocity of 1 along z.

        Call VelocityCylinder module to measure,
        using a bin width of 2, and a grouping per molecule.
        """
        ag_v, bin_volume = circle_of_water_molecules(myvel=np.array([0, 0, 1]),
                                                     bin_width=2,
                                                     radius=rad[0])
        vel = VelocityCylinder(ag_v, vdim=2, bin_width=2,
                               grouping="molecules",
                               refgroup=ag_v).run()
        assert_allclose(vel.results.profile.T[0],
                        vel_array[rad[1]] / bin_volume)
