#!/usr/bin/env python3
"""Tests for the transport modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from create_mda_universe import (
    circle_of_water_molecules,
    line_of_water_molecules,
    )
from numpy.testing import assert_allclose

from maicos import VelocityCylinder, VelocityPlanar


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_TPR, WATER_TRR  # noqa: E402


class TestVelocityPlanar(object):
    """Tests for the velocity planar class."""

    @pytest.fixture()
    def vel_array_1(self):
        """Set velocity array."""
        v_array_1 = []
        # dim = 0:
        v_array_1.append([-0.0019, -0.0029, 0.00058])
        # dim = 1:
        v_array_1.append([-0.0019, -0.0029, 0.00058])
        # dim = 2:
        v_array_1.append([-0.0019, -0.0029, 0.00058])
        return v_array_1

    @pytest.fixture()
    def vel_array_2(self):
        """Set velocity array."""
        v_array_2 = [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]
        return v_array_2

    @pytest.fixture()
    def vel_array_3(self):
        """Set velocity array."""
        v_array_3 = [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]
        return v_array_3

    @pytest.fixture()
    def flux_array_1(self):
        """Set flux array."""
        f_array_1 = [[3.0, 0.0, 0.0],
                     [0.0, 3.0, 0.0],
                     [0.0, 0.0, 3.0]]
        return f_array_1

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_wrong_vdim(self, ag):
        """Test the wrong dimensions for velocity."""
        with pytest.raises(ValueError, match="Velocity dimension can"):
            VelocityPlanar(ag, dim=2, vdim=3)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    @pytest.fixture()
    def test_vel_1(self, ag, dim, vdim, vel_array_1):
        """Test velocity module using WATER_TPR data."""
        vel = VelocityPlanar(ag, dim=dim, vdim=vdim).run()
        assert_allclose(vel.results.profile_mean.mean(),
                        vel_array_1[dim][vdim])

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    def test_vel_molecules(self, dim, vdim, vel_array_2):
        """Test velocity module with 1 water with grouping by molecules."""
        myvel = np.zeros(3)
        myvel[dim] += 1

        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vol = np.prod(ag_v.dimensions[:3])

        vel = VelocityPlanar(ag_v, vdim=vdim, bin_width=10,
                             grouping="molecules").run()

        # Divide by volume for normalization as in module.
        assert_allclose(vel.results.profile_mean.mean(),
                        vel_array_2[dim][vdim] / vol, rtol=1e-6)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    def test_vel_atoms(self, dim, vdim, vel_array_3):
        """Test velocity module with 1 water with grouping by atoms."""
        myvel = np.zeros(3)
        myvel[dim] += 1

        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vol = np.prod(ag_v.dimensions[:3])

        vel = VelocityPlanar(ag_v, vdim=vdim, bin_width=10,
                             grouping="atoms").run()

        # Divide by volume for normalization as in module.
        assert_allclose(vel.results.profile_mean.mean(),
                        vel_array_3[dim][vdim] / vol)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    def test_flux(self, dim, vdim, flux_array_1):
        """Measure flux with 1 water molecules."""
        myvel = np.zeros(3)
        myvel[dim] += 1

        ag_v = line_of_water_molecules(n_molecules=1, myvel=myvel)
        vol = np.prod(ag_v.dimensions[:3])

        vel = VelocityPlanar(ag_v, vdim=vdim, bin_width=10,
                             grouping="atoms", flux=True).run()

        # Divide by volume for normalization as in module.
        assert_allclose(vel.results.profile_mean,
                        flux_array_1[dim][vdim] / vol)


class TestVelocityCylinder(object):
    """Tests for the velocity cylinder class."""

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
