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
from maicos import epsilon_bulk, epsilon_planar, epsilon_cylinder
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from datafiles import WATER_GRO, WATER_TPR, WATER_TRR


class Test_epsilon_bulk(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_broken_molecules(self, ag):
        eps = epsilon_bulk(ag, bpbc=False).run()
        assert_almost_equal(eps.results['eps_mean'], 920.85, decimal=1)

    def test_repaired_molecules(self, ag):
        eps = epsilon_bulk(ag, bpbc=True).run()
        assert_almost_equal(eps.results['eps_mean'], 20.35, decimal=1)

    def test_temperature(self, ag):
        eps = epsilon_bulk(ag, temperature=100).run()
        assert_almost_equal(eps.results['eps_mean'], 59.06, decimal=1)

    def test_output(self, ag):
        with tempdir.in_tempdir():
            eps = epsilon_bulk(ag, save=True).run()
            res = np.loadtxt(eps.output)
            assert_almost_equal(np.hstack(
                [eps.results["eps_mean"], eps.results["eps"]]).T,
                                res,
                                decimal=2)

    def test_output_name(self, ag):
        with tempdir.in_tempdir():
            epsilon_bulk(ag, output="foo", save=True).run()
            open("foo.dat")

    def test_verbose(self, ag):
        epsilon_bulk(ag, verbose=True).run()


class Test_epsilon_planar(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.mark.parametrize('dim, val_perp, val_par',
                             ((0, -0.99, 1026.1), (1, -0.99, 943.2),
                              (2, -0.99, 839.7)))
    def test_broken_molecules(self, ag, dim, val_perp, val_par):
        eps = epsilon_planar(ag, bpbc=False, dim=dim).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), val_perp, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), val_par, decimal=1)

    def test_repaired_molecules(self, ag):
        eps = epsilon_planar(ag, bpbc=True).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), -0.98, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), 9.4, decimal=1)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        eps = epsilon_planar(ag_single_frame, binwidth=0.1).run()
        # Divide by 10: Å -> nm
        n_bins = ag_single_frame.universe.dimensions[dim] / 10 // 0.1
        assert_almost_equal(eps.results["z"][1] - eps.results["z"][0],
                            0.1,
                            decimal=2)
        assert_equal(len(eps.results["z"]), n_bins)

    def test_output(self, ag_single_frame):
        with tempdir.in_tempdir():
            eps = epsilon_planar(ag_single_frame, save=True).run()
            res_perp = np.loadtxt("{}_perp.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_perp"][:, 0],
                                res_perp[:, 1],
                                decimal=1)
            res_par = np.loadtxt("{}_par.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_par"][:, 0],
                                res_par[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame):
        with tempdir.in_tempdir():
            epsilon_planar(ag_single_frame, output_prefix="foo",
                           save=True).run()
            open("foo_perp.dat")
            open("foo_par.dat")

    def test_verbose(self, ag_single_frame):
        epsilon_planar(ag_single_frame, verbose=True).run()


class Test_epsilon_cylinder(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_broken_molecules(self, ag):
        eps = epsilon_cylinder(ag, bpbc=False).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 1365.9, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -9.97, decimal=1)

    def test_repaired_molecules(self, ag):
        eps = epsilon_cylinder(ag, bpbc=True).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 19.9, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -9.79, decimal=1)

    def test_binwidth(self, ag_single_frame):
        eps = epsilon_cylinder(ag_single_frame, binwidth=0.1).run()
        # Divide by 10: Å -> nm
        n_bins = np.ceil(ag_single_frame.universe.dimensions.min() / 10 / 2 /
                         0.1)
        assert_almost_equal(eps.results["r"][1] - eps.results["r"][0],
                            0.1,
                            decimal=2)
        assert_equal(len(eps.results["r"]), n_bins)

    def test_output(self, ag_single_frame):
        with tempdir.in_tempdir():
            eps = epsilon_cylinder(ag_single_frame, save=True).run()
            res_ax = np.loadtxt("{}_ax.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_ax"], res_ax[:, 1], decimal=1)
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_rad"],
                                res_rad[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame):
        with tempdir.in_tempdir():
            epsilon_cylinder(ag_single_frame, output_prefix="foo",
                             save=True).run()
            open("foo_ax.dat")
            open("foo_rad.dat")

    def test_verbose(self, ag_single_frame):
        epsilon_cylinder(ag_single_frame, verbose=True).run()
