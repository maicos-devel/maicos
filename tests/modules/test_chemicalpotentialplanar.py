#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ChemicalPotentialPlanar module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_densityplanar import ReferenceAtomGroups

from maicos import ChemicalPotentialPlanar


class TestChemicalPotentialPlanar(ReferenceAtomGroups):
    """Tests for the ChemicalPotentialPlanar module."""

    def test_mu(self, ag):
        """Test mu."""
        chem_pot = ChemicalPotentialPlanar(ag).run()
        assert_allclose(chem_pot.results["mu"], -19.27, rtol=1e-1)

    def test_mu_error(self, ag):
        """Test mu error."""
        chem_pot = ChemicalPotentialPlanar(ag).run()
        assert_allclose(chem_pot.results["dmu"], 0.08, rtol=1e-1)

    def test_mu_temp(self, ag):
        """Test mu temperature."""
        chem_pot = ChemicalPotentialPlanar(ag, temperature=200).run()
        assert_allclose(chem_pot.results["mu"], -11.8, rtol=1e-1)

    def test_mu_mass(self, ag):
        """Test mu mass."""
        chem_pot = ChemicalPotentialPlanar(ag, mass=40).run()
        assert_allclose(chem_pot.results["mu"], -22.25, rtol=1e-1)

    def test_mu_masses(self, multiple_ags_mu):
        """Test mu masses."""
        chem_pot = ChemicalPotentialPlanar(multiple_ags_mu,
                                           mass=[18, 25, 40],
                                           zpos=40).run()
        assert_allclose(chem_pot.results["mu"], [-19.3, -20.5, -22.3],
                        rtol=1e-1)

    def test_mu_zpos(self, ag):
        """Test mu z position."""
        chem_pot = ChemicalPotentialPlanar(ag, zpos=22).run()
        assert_allclose(chem_pot.results["mu"], -19.27, rtol=1e-1)

    def test_mu_no_mass(self, ag_no_masses):
        """Test mu no mass."""
        with pytest.raises(ValueError):
            ChemicalPotentialPlanar(ag_no_masses).run()

    def test_mu_two_residues(self, multiple_res_ag):
        """Test mu two residues."""
        chem_pot = ChemicalPotentialPlanar(multiple_res_ag,
                                           zpos=0).run()
        assert_allclose(chem_pot.results["mu"], -33.6, rtol=1e-1)

    def test_mu_multiple_ags(self, multiple_ags_mu):
        """Test mu multiples ags."""
        chem_pot = ChemicalPotentialPlanar(multiple_ags_mu,
                                           zpos=40).run()
        assert_allclose(chem_pot.results["mu"], [-19.3, -np.inf, -30.0],
                        rtol=1e-1)

    def test_mu_mult_res_mult_atoms_ag(self, mult_res_mult_atoms_ag):
        """Test output multiple res multiple atoms ag."""
        with pytest.raises(NotImplementedError):
            ChemicalPotentialPlanar(mult_res_mult_atoms_ag,
                                    zpos=40).run()

    def test_output(self, ag_single_frame, tmpdir):
        """Test output."""
        with tmpdir.as_cwd():
            chem_pot = ChemicalPotentialPlanar(ag_single_frame).run()
            chem_pot.save()
            res = np.loadtxt(chem_pot.muout)
            assert_allclose(chem_pot.results["mu"], res[0], rtol=1e-2)

    def test_output_name(self, ag_single_frame, tmpdir):
        """Test output name."""
        with tmpdir.as_cwd():
            chem_pot = ChemicalPotentialPlanar(
                ag_single_frame,
                muout="foo_mu").run()
            chem_pot.save()
            open("foo_mu.dat")
