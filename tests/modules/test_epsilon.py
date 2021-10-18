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

    def test_output(self, ag, tmpdir):
        with tmpdir.as_cwd():
            eps = epsilon_bulk(ag, save=True).run()
            res = np.loadtxt(eps.output)
            assert_almost_equal(np.hstack(
                [eps.results["eps_mean"], eps.results["eps"]]).T,
                                res,
                                decimal=2)

    def test_output_name(self, ag, tmpdir):
        with tmpdir.as_cwd():
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
                             ((0, 0.075, -19.18), 
                              (1, 0.280, -8.1),
                              (2, 0.109, -15.9)))
    def test_broken_molecules(self, ag, dim, val_perp, val_par):
        eps = epsilon_planar(ag, bpbc=False, dim=dim).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), val_perp, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), val_par, decimal=1)

    def test_repaired_molecules(self, ag):
        eps = epsilon_planar(ag, bpbc=True).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), 0.43, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), 1.94, decimal=1)

    def test_output(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            eps = epsilon_planar(ag_single_frame, save=True).run()
            res_perp = np.loadtxt("{}_perp.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_perp"][:, 0],
                                res_perp[:, 1],
                                decimal=1)
            res_par = np.loadtxt("{}_par.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_par"][:, 0],
                                res_par[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
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

    def test_output(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            eps = epsilon_cylinder(ag_single_frame, save=True).run()
            res_ax = np.loadtxt("{}_ax.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_ax"], res_ax[:, 1], decimal=1)
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_rad"],
                                res_rad[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            epsilon_cylinder(ag_single_frame, output_prefix="foo",
                             save=True).run()
            open("foo_ax.dat")
            open("foo_rad.dat")

    def test_verbose(self, ag_single_frame):
        epsilon_cylinder(ag_single_frame, verbose=True).run()
