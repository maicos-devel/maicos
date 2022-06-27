#!/usr/bin/env python3
"""Tests for the epsilon modules."""
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
from datafiles import WATER_GRO, WATER_TPR, WATER_TRR
from numpy.testing import assert_almost_equal, assert_equal

from maicos import DielectricSpectrum, EpsilonCylinder, EpsilonPlanar


class TestEpsilonPlanar(object):
    """Tests for the EpsilonPlanar class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            eps = EpsilonPlanar(ag_single_frame)
            eps.run()
            eps.save()
            res_perp = np.loadtxt("{}_perp.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_perp"][:, 0],
                                res_perp[:, 1],
                                decimal=1)
            res_par = np.loadtxt("{}_par.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_par"][:, 0],
                                res_par[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            eps = EpsilonPlanar(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_perp.dat")
            open("foo_par.dat")

    def test_xy_vac(self, ag):
        """Tests for conditions xy & vac when True."""
        eps1 = EpsilonPlanar(ag, xy=True)
        eps1.run()
        k1 = np.mean(eps1.results.eps_perp)
        eps2 = EpsilonPlanar(ag, xy=True, vac=True)
        eps2.run()
        k2 = np.mean(eps2.results.eps_perp)
        assert_almost_equal((k1/k2), 1.5, decimal=1)

    def test_sym(self, ag):
        """Test for symmetric case."""
        eps = EpsilonPlanar(ag, sym=True)
        eps.run()
        assert_almost_equal(np.mean(eps.results.eps_perp), -1.01, decimal=2)


class TestEpsilonCylinder(object):
    """Tests for the EpsilonCylinder class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_radius(self, ag):
        """Tests radius set."""
        eps = EpsilonCylinder(ag, make_whole=False, radius=50)
        eps.run(start=0, stop=1)
        assert eps.radius == 50

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        eps = EpsilonCylinder(ag).run(stop=1)
        assert not np.isnan(eps.results.eps_rad).any()
        assert not np.isnan(eps.results.eps_ax).any()

    def test_radius_box(self, ag):
        """Tests radius taken from box."""
        eps = EpsilonCylinder(ag, make_whole=False)
        eps.run(start=0, stop=1)
        assert eps.radius == ag.universe.dimensions[:2].min() / 2

    def test_broken_molecules(self, ag):
        """Tests broken molecules."""
        eps = EpsilonCylinder(ag, make_whole=False).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 1179.0, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -10, decimal=0)

    def test_repaired_molecules(self, ag):
        """Tests repaired molecules."""
        eps = EpsilonCylinder(ag, make_whole=True).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 1179.6, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -10, decimal=0)

    def test_output(self, ag_single_frame, tmpdir):
        """Tests output."""
        with tmpdir.as_cwd():
            eps = EpsilonCylinder(ag_single_frame)
            eps.run()
            eps.save()
            res_ax = np.loadtxt("{}_ax.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_ax"], res_ax[:, 1], decimal=1)
            res_rad = np.loadtxt("{}_rad.dat".format(eps.output_prefix))
            assert_almost_equal(eps.results["eps_rad"],
                                res_rad[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Tests output name."""
        with tmpdir.as_cwd():
            eps = EpsilonCylinder(ag_single_frame, output_prefix="foo")
            eps.run()
            eps.save()
            open("foo_ax.dat")
            open("foo_rad.dat")

    def test_verbose(self, ag_single_frame):
        """Tests verbose."""
        EpsilonCylinder(ag_single_frame, verbose=True).run()

    def test_length(self, ag):
        """Test refactoring length."""
        eps = EpsilonCylinder(ag, length=100)
        eps.run()
        assert_equal(eps.length, 100)

    def test_variable_binwidth(self, ag):
        """Test variable binwidth."""
        eps = EpsilonCylinder(ag, variable_dr=True)
        eps.run()
        assert_almost_equal(np.std(eps.dr), 0.44, decimal=2)

    def test_singleline(self, ag):
        """Test for single line 1D case."""
        eps = EpsilonCylinder(ag, single=True)
        eps.run()
        assert_almost_equal(np.mean(eps.results.eps_ax), 1282, decimal=0)


class TestDielectricSpectrum(object):
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_output_name(self, ag, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag)
            ds.run()
            ds.save()
            open("susc.dat")
            open("P_tseries.npy")
            open("tseries.npy")
            open("V.txt")

    def test_output_name_prefix(self, ag, tmpdir):
        """Test output name with custom prefix."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag, output_prefix="foo")
            ds.run()
            ds.save()
            open("foo_susc.dat")
            open("foo_P_tseries.npy")
            open("foo_tseries.npy")
            open("foo_V.txt")

    def test_output_name_binned(self, ag, tmpdir):
        """Test output name of binned data."""
        """
        The parameters are not meant to be sensible,
        but just to force the binned output.
        """
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag, bins=5, binafter=0, segs=5)
            ds.run()
            ds.save()
            open("susc.dat")
            open("susc_binned.dat")
            open("P_tseries.npy")
            open("tseries.npy")
            open("V.txt")

    def test_output(self, ag, tmpdir):
        """Test output values by comparing with magic numbers."""
        with tmpdir.as_cwd():
            ds = DielectricSpectrum(ag)
            ds.run()

            V = 1559814.4
            nu = [0., 0.2, 0.5, 0.7, 1.]
            susc = [27.5 + 0.j, 2.9 + 22.3j, -5.0 + 3.6j,
                    -0.5 + 10.7j, -16.8 + 3.5j]
            dsusc = [3.4 + 0.j, 0.4 + 2.9j, 1.0 + 0.5j, 0.3 + 1.5j, 2.0 + 0.6j]

            assert_almost_equal(ds.V, V, decimal=1)
            assert_almost_equal(ds.results.nu, nu, decimal=1)
            assert_almost_equal(ds.results.susc, susc, decimal=1)
            assert_almost_equal(ds.results.dsusc, dsusc, decimal=1)

    def test_binning(self, ag):
        """Test binning & seglen case."""
        ds = DielectricSpectrum(ag, nobin=False, segs=2, bins=49)
        ds.run()
        assert_almost_equal(np.mean(ds.results.nu_binned), 0.57, decimal=2)
        ds.save()
