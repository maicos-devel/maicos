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

    @pytest.fixture()
    def single_dipole(self):
        """Single dipole atomgroup.

        Create MDA universe with a single dipole molecule
        inside a 1x1x1nm box cubic box.
        """
        u = mda.Universe.empty(2,
                               n_residues=1,
                               atom_resindex=[0, 0],
                               n_segments=1,
                               residue_segindex=[0],
                               trajectory=True)
        positions = np.array([[4, 4, 4],
                              [6, 6, 6]])
        charges = np.array([-0.5, 0.5])
        dimensions = np.array([10, 10, 10, 90, 90, 90])
        u.atoms.positions = positions
        u.add_TopologyAttr('charges', charges)
        u.add_TopologyAttr('resid', [0])
        u.add_TopologyAttr('bonds', [(0, 1)])
        u.add_TopologyAttr('masses', [1, 1])
        u.dimensions = dimensions
        return u.atoms

    @pytest.mark.parametrize('binwidth, bins_par, bins_perp',
                             ((0.5, [0., 0.01992], [0.005, 0]),
                              (0.1, [0, 0, 0, 0, 0, 0.0996, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0.005, 0.005, 0, 0, 0, 0]))
                             )
    def test_single_frame(self, single_dipole, binwidth, bins_par, bins_perp):
        """Test physical quantities."""
        eps = EpsilonPlanar(single_dipole, binwidth=binwidth)
        eps.run()
        assert eps.M_par[0] == 2.
        assert eps.M_perp[0] == 1.
        assert np.all(eps.m_par[:, 0, 0] == bins_par)
        assert np.all(eps.m_perp[:, 0, 0] == bins_perp)

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

    def test_radius(self, ag):
        """Tests radius set."""
        eps = EpsilonCylinder(ag, make_whole=False, radius=5)
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
