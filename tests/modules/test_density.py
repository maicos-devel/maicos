#!/usr/bin/env python3
"""Tests for the density modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.datafiles import TPR, TRR
from numpy.testing import assert_allclose, assert_equal

from maicos.modules import density


sys.path.append("..")
from data import (  # noqa: E402
    MICA_TPR,
    MICA_XTC,
    SALT_WATER_GRO,
    SALT_WATER_TPR,
    WATER_GRO,
    WATER_TPR,
    WATER_TRR,
    )


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

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


class TestTemperaturProfile(ReferenceAtomGroups):
    """Tests for the density.TemperaturePlanar class."""

    def test_multiple(self, multiple_ags):
        """Test temperature."""
        temp = density.TemperaturePlanar(multiple_ags).run()
        assert_allclose(temp.results.profile_mean[40], [223, 259], rtol=1e1)

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dens(self, ag, dim):
        """Test mean temperature."""
        dens = density.TemperaturePlanar(ag, dim=dim).run()
        assert_allclose(dens.results.profile_mean.mean(), 291.6, rtol=1e1)


class TestChemicalPotentialPlanar(ReferenceAtomGroups):
    """Tests for the density.ChemicalPotentialPlanar class."""

    def test_mu(self, ag):
        """Test mu."""
        chem_pot = density.ChemicalPotentialPlanar(ag).run()
        assert_allclose(chem_pot.results["mu"], -19.27, rtol=1e-1)

    def test_mu_error(self, ag):
        """Test mu error."""
        chem_pot = density.ChemicalPotentialPlanar(ag).run()
        assert_allclose(chem_pot.results["dmu"], 0.08, rtol=1e-1)

    def test_mu_temp(self, ag):
        """Test mu temperature."""
        chem_pot = density.ChemicalPotentialPlanar(ag, temperature=200).run()
        assert_allclose(chem_pot.results["mu"], -11.8, rtol=1e-1)

    def test_mu_mass(self, ag):
        """Test mu mass."""
        chem_pot = density.ChemicalPotentialPlanar(ag, mass=40).run()
        assert_allclose(chem_pot.results["mu"], -22.25, rtol=1e-1)

    def test_mu_masses(self, multiple_ags_mu):
        """Test mu masses."""
        chem_pot = density.ChemicalPotentialPlanar(multiple_ags_mu,
                                                   mass=[18, 25, 40],
                                                   zpos=40).run()
        assert_allclose(chem_pot.results["mu"], [-19.3, -20.5, -22.3],
                        rtol=1e-1)

    def test_mu_zpos(self, ag):
        """Test mu z position."""
        chem_pot = density.ChemicalPotentialPlanar(ag, zpos=22).run()
        assert_allclose(chem_pot.results["mu"], -19.27, rtol=1e-1)

    def test_mu_no_mass(self, ag_no_masses):
        """Test mu no mass."""
        with pytest.raises(ValueError):
            density.ChemicalPotentialPlanar(ag_no_masses).run()

    def test_mu_two_residues(self, multiple_res_ag):
        """Test mu two residues."""
        chem_pot = density.ChemicalPotentialPlanar(multiple_res_ag,
                                                   zpos=0).run()
        assert_allclose(chem_pot.results["mu"], -33.6, rtol=1e-1)

    def test_mu_multiple_ags(self, multiple_ags_mu):
        """Test mu multiples ags."""
        chem_pot = density.ChemicalPotentialPlanar(multiple_ags_mu,
                                                   zpos=40).run()
        assert_allclose(chem_pot.results["mu"], [-19.3, -np.inf, -30.0],
                        rtol=1e-1)

    def test_mu_mult_res_mult_atoms_ag(self, mult_res_mult_atoms_ag):
        """Test output multiple res multiple atoms ag."""
        with pytest.raises(NotImplementedError):
            density.ChemicalPotentialPlanar(mult_res_mult_atoms_ag,
                                            zpos=40).run()

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            chem_pot = density.ChemicalPotentialPlanar(ag_single_frame).run()
            chem_pot.save()
            res = np.loadtxt(chem_pot.muout)
            assert_allclose(chem_pot.results["mu"], res[0], rtol=1e-2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            chem_pot = density.ChemicalPotentialPlanar(
                ag_single_frame,
                muout="foo_mu").run()
            chem_pot.save()
            open("foo_mu.dat")


class TestDensityPlanar(ReferenceAtomGroups):
    """Tests for the density.DensityPlanar class."""

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', [0.8129, 0.0122]),
                              ('number', [0.1785, 0.0014]),
                              ('charge', [-1.5E-3, 0])))
    def test_multiple(self, multiple_ags, dens_type, mean):
        """Test multiple."""
        dens = density.DensityPlanar(multiple_ags, dens=dens_type).run()
        assert_allclose(dens.results.profile_mean[40], mean,
                        rtol=1e-1, atol=1e-1)

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 0.594), ('number', 0.099),
                              ('charge', 0)))
    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_dens(self, ag, dens_type, mean, dim):
        """Test density."""
        dens = density.DensityPlanar(ag, dens=dens_type, dim=dim).run()
        assert_allclose(dens.results.profile_mean.mean(), mean,
                        rtol=1e-1, atol=1e-8)

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        dens = density.DensityPlanar(ag).run(stop=1)
        assert not np.isnan(dens.results.profile_mean).any()

    @pytest.mark.parametrize('dim', (0, 1, 2))
    def test_binwidth(self, ag_single_frame, dim):
        """Test bin width."""
        dens = density.DensityPlanar(ag_single_frame, binwidth=1,
                                     dim=dim).run()
        n_bins = np.ceil(ag_single_frame.universe.dimensions[dim]) / 1
        assert_allclose(dens.results["z"][1] - dens.results["z"][0],
                        1,
                        rtol=1e-1)
        assert_equal(len(dens.results["z"]), n_bins)

    def test_comshift(self, mica_water):
        """Test comshift."""
        dens = density.DensityPlanar(mica_water, refgroup=mica_water).run()
        assert_allclose(dens.results['profile_mean'][20], 0.581, rtol=1e-1)

    def test_comshift_z2(self, mica_water):
        """Test comshift with an additional shift by z/2."""
        mica_water.atoms.translate(
            (0, 0, mica_water.universe.dimensions[2] / 2))
        dens = density.DensityPlanar(mica_water, refgroup=mica_water).run()
        assert_allclose(dens.results['profile_mean'][20], 0.56, rtol=1e-1)

    def test_comshift_over_boundaries(self, mica_water, mica_surface):
        """Test comshift over box boundaries."""
        dens = density.DensityPlanar(mica_water, refgroup=mica_surface).run()
        assert_allclose(dens.results['profile_mean'][20], 0.0, rtol=1e-1)


class TestDensityCylinder(object):
    """Tests for the density.DensityCylinder class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('dens_type, mean',
                             (('mass', 0.59), ('number', 0.099),
                              ('charge', -1e-4)))
    def test_dens(self, ag, dens_type, mean):
        """Test density."""
        dens = density.DensityCylinder(ag, dens=dens_type).run()
        assert_allclose(dens.results.profile_mean.mean(), mean,
                        atol=1e-4, rtol=1e-2)
