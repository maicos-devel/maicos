#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import MDAnalysis as mda
import pytest

from maicos import velocity
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import WATER_TPR, WATER_TRR


class Test_velocity(object):

    @pytest.fixture()
    def vel_array(self):
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
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    @pytest.mark.parametrize('vdim', (0, 1, 2))
    def test_dens(self, ag, dim, vdim, vel_array):
        vel = velocity(ag, dim=dim, vdim=vdim).run()
        assert_almost_equal(vel.results['v'].mean(),
                            vel_array[dim][vdim],
                            decimal=0)

    def test_broken_molecules(self, ag):
        vel = velocity(ag, bpbc=False).run()
        assert_almost_equal(vel.results['v'].mean(), -0.16, decimal=1)

    def test_repaired_molecules(self, ag):
        vel = velocity(ag, bpbc=True).run()
        assert_almost_equal(vel.results['v'].mean(), -0.26, decimal=1)

    def test_output(self, ag, tmpdir):
        with tmpdir.as_cwd():
            vel = velocity(ag, end=20, save=True, mu=True).run()
            res_vel = np.loadtxt("vel_{}.dat".format(vel.output_suffix))
            assert_almost_equal(vel.results["v"], res_vel[:, 1], decimal=2)

    def test_output_name(self, ag, tmpdir):
        with tmpdir.as_cwd():
            velocity(ag, output_suffix="foo", end=20, save=True).run()
            open("vel_foo.dat")

    def test_verbose(self, ag):
        velocity(ag, verbose=True).run()
