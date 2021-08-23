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

from maicos import density_planar, density_cylinder
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from datafiles import WATER_GRO, WATER_TPR, WATER_TRR

from MDAnalysisTests.datafiles import TPR, XTC, TRR


water_chemical_potential = -19.27


class Test_density_planar(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.fixture()
    def multiple_ags(self):
        u = mda.Universe(TPR, TRR)
        return [u.select_atoms("resname SOL"), u.select_atoms("resname MET")]

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', [1241.7,   18.4]), ('number', [167.4,   1.5]),
                              ('charge', [0.9, 0.05]), ('temp', [227.2, 282.9])))
    def test_multiple(self, multiple_ags, dens_type, mean):
        dens = density_planar(multiple_ags, dens=dens_type).run()
        assert_almost_equal(dens.results['dens_mean'][40], mean, decimal=1)

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 987.9), ('number', 99.1),
                              ('charge', 0.0), ('temp', 291.6)))
    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dens(self, ag, dens_type, mean, dim):
        dens = density_planar(ag, dens=dens_type, dim=dim).run()
        assert_almost_equal(dens.results['dens_mean'].mean(), mean, decimal=0)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        dens = density_planar(ag_single_frame, binwidth=0.1, dim=dim).run()
        # Divide by 10: Å -> nm
        n_bins = np.ceil(ag_single_frame.universe.dimensions[dim]) / 10 / 0.1
        assert_almost_equal(dens.results["z"][1] - dens.results["z"][0],
                            0.1,
                            decimal=2)
        assert_equal(len(dens.results["z"]), n_bins)

    def test_mu(self, ag):
        dens = density_planar(ag, mu=True).run()
        assert_almost_equal(dens.results["mu"],
                            water_chemical_potential,
                            decimal=1)

    def test_mu_temp(self, ag):
        dens = density_planar(ag, mu=True, temperature=200).run()
        assert_almost_equal(dens.results["mu"], -11.8, decimal=1)

    def test_mu_zpos(self, ag):
        dens = density_planar(ag, mu=True, zpos=2.2).run()
        assert_almost_equal(dens.results["mu"],
                            water_chemical_potential,
                            decimal=1)

    def test_mu_not_mass(self, ag):
        with pytest.raises(ValueError):
            density_planar(ag, mu=True, dens="number").run()

    def test_mu_two_groups(self, ag):
        with pytest.warns(UserWarning):
            density_planar([ag, ag], mu=True).run()

    def test_output(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            dens = density_planar(ag_single_frame, save=True, mu=True).run()
            res_dens = np.loadtxt(dens.output)
            res_mu = np.loadtxt(dens.muout)
            assert_almost_equal(dens.results["dens_mean"][:, 0],
                                res_dens[:, 1],
                                decimal=2)
            assert_almost_equal(dens.results["mu"], res_mu[0], decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            density_planar(ag_single_frame,
                           output="foo",
                           muout="foo_mu",
                           save=True,
                           mu=True).run()
            open("foo.dat")
            open("foo_mu.dat")

    def test_verbose(self, ag):
        density_planar(ag, verbose=True).run()

    def test_dens_type(self, ag):
        with pytest.raises(ValueError):
            density_planar(ag, dens="foo").run()


class Test_density_cylinder(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 980.7), ('number', 99.1),
                              ('charge', 0.0), ('temp', 291.6)))
    def test_dens(self, ag, dens_type, mean):
        dens = density_cylinder(ag, dens=dens_type).run()
        assert_almost_equal(dens.results['dens_mean'].mean(), mean, decimal=0)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        dens = density_cylinder(ag_single_frame, binwidth=0.1, dim=dim).run()
        # Divide by 10: Å -> nm
        odims = np.roll(np.arange(3), -dim)[1:]
        n_bins = ag_single_frame.universe.dimensions[odims].min() / 20 / 0.1
        assert_almost_equal(dens.results["r"][1] - dens.results["r"][0],
                            0.1,
                            decimal=2)
        assert_equal(len(dens.results["r"]), np.ceil(n_bins))

    def test_no_center_group(self, ag_single_frame):
        with pytest.raises(RuntimeError):
            density_cylinder(ag_single_frame, center="name foo").run()

    def test_output(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            dens = density_planar(ag_single_frame, save=True).run()
            res = np.loadtxt(dens.output)
            assert_almost_equal(dens.results["dens_mean"][:, 0],
                                res[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        with tmpdir.as_cwd():
            density_planar(ag_single_frame, output="foo", save=True).run()
            open("foo.dat")

    def test_verbose(self, ag_single_frame):
        density_planar(ag_single_frame, verbose=True).run()
