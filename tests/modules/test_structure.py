#!/usr/bin/env python3
# coding: utf8

import MDAnalysis as mda
import pytest

from MDAnalysisTests import tempdir
from maicos import velocity
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import WATER_TPR, WATER_TRR


class Test_velocity(object):

    @pytest.fixture()
    def vel_array(self):
        v_array = []
        # dim = 0:
        v_array.append([8.33, -3.9, 12.98])
        # dim = 1:
        v_array.append([2.44, -3.51, -6.16])
        # dim = 2:
        v_array.append([-4.70, 1.34, 26.46])
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
        assert_almost_equal(vel.results['v'].mean(), -2.9, decimal=1)

    def test_repaired_molecules(self, ag):
        vel = velocity(ag, bpbc=True).run()
        assert_almost_equal(vel.results['v'].mean(), -4.70, decimal=1)

    def test_output(self, ag):
        with tempdir.in_tempdir():
            vel = velocity(ag, end=20, save=True, mu=True).run()
            res_vel = np.loadtxt("vel_{}.dat".format(vel.output_suffix))
            assert_almost_equal(vel.results["v"], res_vel[:, 1], decimal=2)

    def test_output_name(self, ag):
        with tempdir.in_tempdir():
            velocity(ag, output_suffix="foo", end=20, save=True).run()
            open("vel_foo.dat")

    def test_verbose(self, ag):
        velocity(ag, verbose=True).run()
