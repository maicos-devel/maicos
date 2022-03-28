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
from numpy.testing import assert_almost_equal

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

    @pytest.mark.parametrize('dim, val_perp, val_par',
                             ((0, -0.2, 52.7),
                              (1, -0.248, 43.2),
                              (2, -0.22, 37.0)))
    def test_broken_molecules(self, ag, dim, val_perp, val_par):
        """Tests broken molecules."""
        eps = EpsilonPlanar(ag, make_whole=False, dim=dim).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), val_perp, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), val_par, decimal=1)

    def test_repaired_molecules(self, ag):
        """Tests repaired molecules."""
        eps = EpsilonPlanar(ag, make_whole=True).run()
        assert_almost_equal(eps.results['eps_perp'].mean(), -0.43, decimal=1)
        assert_almost_equal(eps.results['eps_par'].mean(), 0.32, decimal=1)

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

    def test_broken_molecules(self, ag):
        """Tests broken molecules."""
        eps = EpsilonCylinder(ag, make_whole=False).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 1365.9, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -9.97, decimal=1)

    def test_repaired_molecules(self, ag):
        """Tests repaired molecules."""
        eps = EpsilonCylinder(ag, make_whole=True).run()
        assert_almost_equal(eps.results['eps_ax'].mean(), 19.9, decimal=1)
        assert_almost_equal(eps.results['eps_rad'].mean(), -9.79, decimal=1)

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


class TestDielectricSpectrum(object):
    """Tests for the DielectricSpectrum class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('plotformat', ["pdf", "png", "jpg", "eps"])
    def test_plotformat(self, ag, plotformat, tmpdir):
        """Test plot format."""
        with tmpdir.as_cwd():
            DielectricSpectrum(ag, plotformat=plotformat,
                               output_prefix='test_it').run()
            assert open('test_it_susc_log.' + plotformat)
            assert open('test_it_susc_linlog.' + plotformat)

    def test_plotformat_wrong(self, ag):
        """Test plot format wrong."""
        with pytest.raises(ValueError,
                           match="Invalid choice for plotformat: 'foo'"):
            DielectricSpectrum(ag, plotformat="foo").run()
