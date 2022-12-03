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
    def vel_array_4(self):
        """Set velocity array."""
        # average velocity of 1 in the third bin
        v_array_1 = np.zeros(5)
        v_array_1[2] += 1
        return v_array_1

    @pytest.fixture()
    def bin_volume_1(self):
        """Set the volume of the bin."""
        # estimate the volume
        _rmax = 10
        _rmin = 0
        _zmax = 20
        _zmin = 0
        _n_bins = 5
        _bin_edges = np.linspace(_rmin, _rmax, _n_bins + 1, endpoint=True)
        _bin_area = np.pi * np.diff(_bin_edges ** 2)
        _L = _zmax - _zmin
        return _bin_area * _L

    def test_vel_cylinder(self, vel_array_4, bin_volume_1):
        """Test velocity module with 10 waters molecules in circle."""
        ag_v = circle_of_water_molecules(myvel=np.array([0, 0, 1]))

        vel = VelocityCylinder(ag_v, vdim=2, bin_width=2,
                               grouping="molecules",
                               refgroup=ag_v).run()

        assert_allclose(vel.results.profile_mean.T[0],
                        vel_array_4 / bin_volume_1)
