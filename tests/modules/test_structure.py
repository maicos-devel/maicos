#!/usr/bin/env python3
"""Tests for the structure modules."""
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
from datafiles import AIRWATER_TPR, AIRWATER_TRR, WATER_TPR, WATER_TRR
from numpy.testing import assert_almost_equal

from maicos import Diporder, Saxs


class TestSaxs(object):
    """Tests for the Saxs class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_saxs(sef, ag):
        """Test Saxs."""
        Saxs(ag, endq=20).run(stop=5)


class TestDiporder(object):
    """Tests for the Diporder class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary."""
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P0"] = np.array([0, 0, 0.01, 0])
        res[0]["cos_theta"] = np.array([0, 0, 0.01, 0])
        res[0]["cos_2_theta"] = np.array([0.34, 0.35, 0.35, 0.35])
        res[0]["rho"] = np.array([14.75, 15., 14.78, 14.73])

        # y-direction
        res[1] = {}
        res[1]["P0"] = np.array([-0.01, 0, 0, 0])
        res[1]["cos_theta"] = np.array([-0.01, -0.01, 0, -0.01])
        res[1]["cos_2_theta"] = np.array([0.34, 0.34, 0.34, 0.35])
        res[1]["rho"] = np.array([14.85, 14.81, 14.8, 14.79])

        # z-direction
        res[2] = {}
        res[2]["P0"] = np.array([-0.01, 0.03, 0., -0.01, -0.,
                                 -0.03, 0.01, 0., 0., 0., 0., 0.])
        res[2]["cos_theta"] = np.array(
            [-0.02, 0.02, 0., -0., -0., -0.02, 0.03, 0., 0., 0., 0., 0.])
        res[2]["cos_2_theta"] = np.array([0.22, 0.30, 0.33, 0.33, 0.33, 0.30,
                                          0.23, 0., 0., 0., 0., 0.])
        res[2]["rho"] = np.array(
            [5.96, 32.38, 33.26, 33.1, 33.38, 32.63, 7.05, 0., 0., 0., 0., 0.])

        return res

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_Diporder(self, ag, dim, result_dict):
        """Test Diporder."""
        dip = Diporder(ag, binwidth=0.5, dim=dim).run()
        assert_almost_equal(dip.results['P0'],
                            result_dict[dim]['P0'],
                            decimal=2)
        assert_almost_equal(dip.results['cos_theta'],
                            result_dict[dim]['cos_theta'],
                            decimal=2)
        assert_almost_equal(dip.results['cos_2_theta'],
                            result_dict[dim]['cos_2_theta'],
                            decimal=2)
        assert_almost_equal(dip.results['rho'],
                            result_dict[dim]['rho'],
                            decimal=2)

    def test_broken_molecules(self, ag):
        """Test broken molecules."""
        dip = Diporder(ag, make_whole=False).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.05, decimal=2)

    def test_repaired_molecules(self, ag):
        """Test repaired molecules."""
        dip = Diporder(ag, make_whole=True).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.00, decimal=2)

    def test_output(self, ag, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            dip = Diporder(ag, end=20)
            dip.run()
            dip.save()
            res_dip = np.loadtxt(dip.output)
            assert_almost_equal(dip.results["P0"], res_dip[:, 1], decimal=2)

    def test_Lz(self, ag):
        """Test Lz."""
        dip = Diporder(ag, bpbc=False).run()
        Lz = ag.universe.trajectory.n_frames * ag.universe.dimensions[2]
        assert dip.Lz == Lz

    def test_output_name(self, ag, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            dip = Diporder(ag, output="foo.dat", end=20)
            dip.run()
            dip.save()
            open("foo.dat")

    def test_verbose(self, ag):
        """Test verbose."""
        Diporder(ag, verbose=True).run()
