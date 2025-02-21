#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the KineticEnergy class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_raises

from maicos import KineticEnergy

sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NVE, WATER_TRR_NVE  # noqa: E402
from util import line_of_water_molecules  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NVE, WATER_TRR_NVE)
        return u.atoms


class TestKineticEnergy(ReferenceAtomGroups):
    """Tests for the KineticEnergy class."""

    def test_ke_trans_trajectory(self, ag):
        """Test translational kinetic energy."""
        ke = KineticEnergy(ag, refpoint="com").run(stop=1)
        assert_allclose(ke.results.trans, 1905.26, rtol=1e-2)

    def test_ke_trans_trajectory_save(self, ag, monkeypatch, tmp_path):
        """Test translational kinetic energy.

        Save the result in a text file, and assert that the results printed in the file
        is correct.
        """
        monkeypatch.chdir(tmp_path)

        ke = KineticEnergy(ag, refpoint="COM").run(stop=1)
        ke.save()
        saved = np.loadtxt("ke.dat")
        assert_allclose(saved[1], 1905.26, rtol=1e-2)

    def test_ke_rot(self, ag):
        """Test rotational kinetic energy."""
        ke = KineticEnergy(ag).run(stop=1)
        assert_allclose(ke.results.rot, 1898.81, rtol=1e-2)

    def test_prepare(self, ag):
        """Test Value error when refpoint is not COM or COC."""
        kem = KineticEnergy(ag, refpoint="OH")
        with assert_raises(ValueError):
            kem.run()

    def test_ke_rot_COC(self, ag):
        """Test rotational KE COC."""
        ke = KineticEnergy(ag, refpoint="COC").run(stop=1)
        assert_allclose(ke.results.rot, 584.17, rtol=1e-1)

    @pytest.mark.parametrize("vel", [0.0, 1.0, 2.0])
    def test_ke_single_molecule(self, vel):
        """Test KineticEnergy module using a single molecule.

        Create a universe with one single water molecule with a given velocity of vel
        along z.

        The expected result corresponds to the 0.5*m*v**2 (in kJ/mol) where m is the
        mass of a single water molecule.
        """
        ag = line_of_water_molecules(n_molecules=1, myvel=np.array([0.0, 0.0, vel]))
        vol = np.prod(ag.dimensions[:3])
        ke = KineticEnergy(ag, refpoint="COM").run()

        mass_h2o = 18.0153 / 1000  # kg/mol
        myke = 0.5 * mass_h2o * (vel * 100) ** 2  # kJ/mol (100 = A/ps to m/s)

        assert_allclose(ke.results.trans, myke / vol, rtol=1e-1)
