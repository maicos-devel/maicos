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

from MDAnalysisTests import tempdir
from maicos import diporder
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import AIRWATER_TPR, AIRWATER_TRR


class Test_diporder(object):
    @pytest.fixture()
    def result_dict(self):
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P0"] = np.array([0, 0, 0, 0])
        res[0]["cos_theta"] = np.array([0, 0, 0.01, 0])
        res[0]["cos_2_theta"] = np.array([0.36, 0.37, 0.4, 0.41])
        res[0]["rho"] = np.array([15.01, 14.73, 14.79, 14.73])

        # y-direction
        res[1] = {}
        res[1]["P0"] = np.array([0, 0, 0, 0])
        res[1]["cos_theta"] = np.array([0, 0, -0.01, -0.01])
        res[1]["cos_2_theta"] = np.array([0.41, 0.38, 0.42, 0.47])
        res[1]["rho"] = np.array([14.71, 14.84, 14.88, 14.83])

        # z-direction
        res[2] = {}
        res[2]["P0"] = np.array([0.02, 0, 0, 0, 0, -0.03, 0, 0, 0, 0, 0, 0])
        res[2]["cos_theta"] = np.array(
            [0.02, 0, 0, 0, 0, -0.02, 0.08, 0, 0, 0, 0, -0.06])
        res[2]["cos_2_theta"] = np.array(
            [0.08, 0.05, 0.06, 0.04, 0.05, 0.09, 0.09, 0, 0, 0, 0, 0.05])
        res[2]["rho"] = np.array(
            [21.31, 33.3, 33.38, 33.14, 33.03, 22.94, 0.41, 0, 0, 0, 0, 0.27])

        return res

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_diporder(self, ag, dim, result_dict):
        dip = diporder(ag, binwidth=0.5, dim=dim).run()
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
        dip = diporder(ag, bpbc=False).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.05, decimal=2)

    def test_repaired_molecules(self, ag):
        dip = diporder(ag, bpbc=True).run()
        assert_almost_equal(dip.results['P0'].mean(), 0.00, decimal=2)

    def test_output(self, ag):
        with tempdir.in_tempdir():
            dip = diporder(ag, end=20, save=True).run()
            res_dip = np.loadtxt(dip.output)
            assert_almost_equal(dip.results["P0"], res_dip[:, 1], decimal=2)

    def test_output_name(self, ag):
        with tempdir.in_tempdir():
            diporder(ag, output="foo.dat", end=20, save=True).run()
            open("foo.dat")

    def test_verbose(self, ag):
        diporder(ag, verbose=True).run()
