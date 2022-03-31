#!/usr/bin/env python3
"""Tests for the density modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import os

import MDAnalysis as mda
import numpy as np
import pytest
from datafiles import (
    MICA_TPR,
    MICA_XTC,
    SALT_WATER_GRO,
    SALT_WATER_TPR,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )
from MDAnalysisTests.datafiles import TPR, TRR
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_warns,
    )

from maicos import DensityCylinder, DensityPlanar


class TestDensityPlanar(object):
    """Tests for the DensityPlanar class."""

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
    def multiple_ags(self):
        """Import MDA universe, multiple ags."""
        u = mda.Universe(TPR, TRR)
        return [u.select_atoms("resname SOL"),
                u.select_atoms("resname MET")]

    @pytest.fixture()
    def multiple_ags_mu(self):
        """Import MDA universe, multiple ags mu."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [u.select_atoms("resname SOL"),
                u.select_atoms("resname NA"),
                u.select_atoms("resname CL")]

    @pytest.fixture()
    def mica_water(self):
        """Import MDA universe, water components of a slab system."""
        u = mda.Universe(MICA_TPR, MICA_XTC)
        return u.select_atoms('resname SOL')

    @pytest.fixture()
    def mica_surface(self):
        """Import MDA universe, surface component of a slab system."""
        u = mda.Universe(MICA_TPR, MICA_XTC)
        return u.select_atoms('resname SURF')

    @pytest.fixture()
    def ag_no_masses(self):
        """Atom group with no mass."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        u.del_TopologyAttr('masses')
        return u.atoms

    @pytest.fixture()
    def multiple_res_ag(self):
        """Import MDA universe, multiple resname atom group."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [u.select_atoms("resname NA or resname CL")]

    @pytest.fixture()
    def mult_res_mult_atoms_ag(self):
        """Import MDA universe, multiple resname atom group."""
        u = mda.Universe(SALT_WATER_TPR, SALT_WATER_GRO)
        return [u.select_atoms("resname SOL or resname NA")]

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', [1350, 20]),
                              ('number', [178.5, 1.4]),
                              ('charge', [-1.5, 0.0]),
                              ('temp', [223, np.nan])))
    def test_multiple(self, multiple_ags, dens_type, mean):
        """Test multiple."""
        dens = DensityPlanar(multiple_ags, dens=dens_type).run()
        assert_almost_equal(dens.results['dens_mean'][40], mean, decimal=0)

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 988), ('number', 99.1),
                              ('charge', 0.0), ('temp', 291.6)))
    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dens(self, ag, dens_type, mean, dim):
        """Test density."""
        dens = DensityPlanar(ag, dens=dens_type, dim=dim).run()
        assert_almost_equal(dens.results['dens_mean'].mean(), mean, decimal=0)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        """Test bin width."""
        dens = DensityPlanar(ag_single_frame, binwidth=0.1, dim=dim).run()
        # Divide by 10: Å -> nm
        n_bins = np.ceil(ag_single_frame.universe.dimensions[dim]) / 10 / 0.1
        assert_almost_equal(dens.results["z"][1] - dens.results["z"][0],
                            0.1, decimal=2)
        assert_equal(len(dens.results["z"]), n_bins)

    def test_comshift(self, mica_water):
        """Test comshift."""
        dens = DensityPlanar(mica_water, comgroup=mica_water).run()
        assert_almost_equal(dens.results['dens_mean'][20], 966, decimal=1)

    def test_comshift_z2(self, mica_water):
        """Test comshift with an additional shift by z/2."""
        mica_water.atoms.translate(
            (0, 0, mica_water.universe.dimensions[2] / 2))
        dens = DensityPlanar(mica_water, comgroup=mica_water).run()
        assert_almost_equal(dens.results['dens_mean'][20], 966, decimal=1)

    def test_comshift_over_boundaries(self, mica_water, mica_surface):
        """Test comshift over box boundaries."""
        dens = DensityPlanar(mica_water, comgroup=mica_surface).run()
        assert_almost_equal(dens.results['dens_mean'][20], 0, decimal=1)

    def test_mu(self, ag):
        """Test mu."""
        dens = DensityPlanar(ag, mu=True, dens='number').run()
        assert_almost_equal(dens.results["mu"], -19.27, decimal=1)

    def test_mu_error(self, ag):
        """Test mu error."""
        dens = DensityPlanar(ag, mu=True, dens='number').run()
        assert_almost_equal(dens.results["dmu"], 0.04, decimal=1)

    def test_mu_temp(self, ag):
        """Test mu temperature."""
        dens = DensityPlanar(ag, mu=True, dens='number', temperature=200).run()
        assert_almost_equal(dens.results["mu"], -11.8, decimal=1)

    def test_mu_mass(self, ag):
        """Test mu mass."""
        dens = DensityPlanar(ag, mu=True, dens='number', mass=40).run()
        assert_almost_equal(dens.results["mu"], -22.25, decimal=1)

    def test_mu_masses(self, multiple_ags_mu):
        """Test mu masses."""
        dens = DensityPlanar(multiple_ags_mu, mu=True, dens='number',
                             mass=[18, 25, 40], zpos=4).run()
        assert_almost_equal(dens.results["mu"], [-19.3, -20.5, -22.3],
                            decimal=1)

    def test_mu_zpos(self, ag):
        """Test mu z position."""
        dens = DensityPlanar(ag, mu=True, dens='number', zpos=2.2).run()
        assert_almost_equal(dens.results["mu"], -19.27, decimal=1)

    def test_mu_not_number(self, ag):
        """Test mu not number."""
        with pytest.raises(ValueError):
            DensityPlanar(ag, mu=True, dens='mass').run()

    def test_mu_no_mass(self, ag_no_masses):
        """Test mu no mass."""
        with pytest.raises(ValueError):
            DensityPlanar(ag_no_masses, mu=True, dens='number').run()

    def test_mu_two_residues(self, multiple_res_ag):
        """Test mu two residues."""
        dens = DensityPlanar(multiple_res_ag, mu=True, dens='number',
                             zpos=0).run()
        assert_almost_equal(dens.results["mu"], -33.6, decimal=1)

    def test_mu_multiple_ags(self, multiple_ags_mu):
        """Test mu multiples ags."""
        dens = DensityPlanar(multiple_ags_mu, mu=True, dens='number',
                             zpos=4).run()
        assert_almost_equal(dens.results["mu"], [-19.3, -np.inf, -30.0],
                            decimal=1)

    def test_mu_mult_res_mult_atoms_ag(self, mult_res_mult_atoms_ag):
        """Test output multiple res multiple atoms ag."""
        with pytest.raises(NotImplementedError):
            DensityPlanar(mult_res_mult_atoms_ag, mu=True, dens='number',
                          zpos=4).run()

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            dens = DensityPlanar(ag_single_frame, mu=True, dens='number')
            dens.run()
            dens.save()
            res_dens = np.loadtxt(dens.output)
            res_mu = np.loadtxt(dens.muout)
            assert_almost_equal(dens.results["dens_mean"][:, 0], res_dens[:, 1],
                                decimal=2)
            assert_almost_equal(dens.results["mu"], res_mu[0], decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            dens = DensityPlanar(ag_single_frame,
                                 output="foo",
                                 muout="foo_mu",
                                 mu=True,
                                 dens='number')
            dens.run()
            dens.save()
            open("foo.dat")
            open("foo_mu.dat")

    def test_dens_type(self, ag):
        """Test density type."""
        with pytest.raises(ValueError):
            DensityPlanar(ag, dens="foo").run()
    
    def test_dens_mu(self, ag):
        """Test conditions in Density & mu."""
        with assert_warns(UserWarning):
            assert DensityPlanar(ag, dens=None, mu=True).run()

    def test_dens_mu1(self, ag):
        """Density =! number and mu us True."""
        dens = DensityPlanar(ag, dens="charge", mu=True)
        with assert_raises(ValueError):
            dens.run()

    def test_dens_None(self, ag):
        """Testing density None."""
        dens = DensityPlanar(ag, dens=None)
        dens.run()
        assert_equal(dens.dens, "mass")

    def test_save_n(self, ag):
        """Testing save method."""
        dens = DensityPlanar(ag, dens="number")
        dens.run()
        dens.save()
        assert_equal(os.path.exists("density.dat"), True)

    def test_save_number(self, ag):
        """Testing on density number flag."""
        dens = DensityPlanar(ag, dens="number")
        dens.run()
        dens.save()
        outputf = open("density.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[4], '[nm^(-3)]')

    def test_save_charge(self, ag):
        """Testing on density charge flag."""
        dens = DensityPlanar(ag, dens="charge")
        dens.run()
        dens.save()
        outputf = open("density.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[4] + data.split()[5], "[enm^(-3)]")

    def test_save_temp(self, ag):
        """Test on density temperature flag."""
        dens = DensityPlanar(ag, dens="temp")
        dens.run()
        dens.save()
        outputf = open("density.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[3], '[K]')


class TestDensityCylinder(object):
    """Tests for the DensityCylinder class."""

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

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 980.7), ('number', 99.1),
                              ('charge', 0.0), ('temp', 291.6)))
    def test_dens(self, ag, dens_type, mean):
        """Test density."""
        dens = DensityCylinder(ag, dens=dens_type).run()
        assert_almost_equal(dens.results['dens_mean'].mean(), mean, decimal=0)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        """Test binwidth."""
        dens = DensityCylinder(ag_single_frame, binwidth=0.1, dim=dim).run()
        # Divide by 10: Å -> nm
        odims = np.roll(np.arange(3), -dim)[1:]
        n_bins = ag_single_frame.universe.dimensions[odims].min() / 20 / 0.1
        assert_almost_equal(dens.results["r"][1] - dens.results["r"][0],
                            0.1,
                            decimal=2)
        assert_equal(len(dens.results["r"]), np.ceil(n_bins))

    def test_no_center_group(self, ag_single_frame):
        """Test no center group."""
        with pytest.raises(RuntimeError):
            DensityCylinder(ag_single_frame, center="name foo").run()

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            dens = DensityPlanar(ag_single_frame)
            dens.run()
            dens.save()
            res = np.loadtxt(dens.output)
            assert_almost_equal(dens.results["dens_mean"][:, 0],
                                res[:, 1],
                                decimal=2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            dens = DensityPlanar(ag_single_frame,
                                 output="foo")
            dens.run()
            dens.save()
            open("foo.dat")

    def test_dens_type(self, ag):
        """Testing error."""
        dens = DensityCylinder(ag, dens="dummy")
        with assert_raises(ValueError):
            dens.run()

    def test_dens_radius_length(self, ag):
        """Testing rescaling of density, radius."""
        dens = DensityCylinder(ag, radius=10.0, length=10.0)
        dens.run()
        assert_equal(dens.radius, 1.0)
        assert_equal(dens.length, 1.0)

    def test_dens_cyl_save(self, ag):
        """Testing save method."""
        dens = DensityCylinder(ag)
        dens.run()
        dens.save()
        assert_equal(os.path.exists("density_cylinder.dat"), True)

    def test_dens_cyl_save_charge(self, ag):
        """Testing with density flag charge."""
        dens = DensityCylinder(ag, dens="charge")
        dens.run()
        dens.save()
        outputf = open("density_cylinder.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[4] + data.split()[5], "[enm^(-3)]")

    def test_dens_cyl_save_number(self, ag):
        """Testing with density flag number."""
        dens = DensityCylinder(ag, dens="number")
        dens.run()
        dens.save()
        outputf = open("density_cylinder.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[4], "[nm^(-3)]")

    def test_dens_cyl_save_temp(self, ag):
        """Testing with density flag temperature."""
        dens = DensityCylinder(ag, dens="temp")
        dens.run()
        dens.save()
        outputf = open("density_cylinder.dat", "r")
        data = outputf.readlines()[1]
        assert_equal(data.split()[3], '[K]')
