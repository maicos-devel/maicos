#!/usr/bin/env python3
"""Tests for the transport modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import MDAnalysis as mda
import numpy as np
import pytest
from datafiles import WATER_TPR, WATER_TRR
from numpy.testing import assert_almost_equal

from maicos import Velocity


class TestVelocity(object):
    """Tests for the velocity class."""

    @pytest.fixture()
    def vel_array(self):
        """Set velocity array."""
        v_array = []
        # dim = 0:
        v_array.append([0.46, -0.21, 0.72])
        # dim = 1:
        v_array.append([0.13, -0.19, -0.34])
        # dim = 2:
        v_array.append([-0.26, 1.34, 1.47])
        return v_array

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    def test_dens(self, ag, dim, vdim, vel_array):
        """Test velocity."""
        vel = Velocity(ag, dim=dim, vdim=vdim).run()
        assert_almost_equal(vel.results['v'].mean(),
                            vel_array[dim][vdim],
                            decimal=0)

    def test_broken_molecules(self, ag):
        """Test broken molecules."""
        vel = Velocity(ag, bpbc=False).run()
        assert_almost_equal(vel.results['v'].mean(), -0.0026, decimal=1)

    def test_repaired_molecules(self, ag):
        """Test repaired molecules."""
        vel = Velocity(ag, bpbc=True).run()
        assert_almost_equal(vel.results['v'].mean(), -0.0026, decimal=1)

    def test_output(self, ag, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            vel = Velocity(ag, end=20, save=True, mu=True)
            vel.run()
            vel.save()
            res_vel = np.loadtxt("vel_{}.dat".format(vel.output_suffix))
            assert_almost_equal(vel.results["v"], res_vel[:, 1], decimal=2)

    def test_output_name(self, ag, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            vel = Velocity(ag, output_suffix="foo", end=20, save=True).run()
            vel.run()
            vel.save()
            open("vel_foo.dat")

    def test_verbose(self, ag):
        """Test verbose."""
        Velocity(ag, verbose=True).run()
