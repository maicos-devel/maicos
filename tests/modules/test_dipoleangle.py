#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DipoleAngle class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from create_mda_universe import line_of_water_molecules
from numpy.testing import assert_allclose

from maicos import DipoleAngle


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_GRO, WATER_TPR  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms


class TestDipoleAngle(ReferenceAtomGroups):
    """Tests for the DipoleAngle class."""

    def test_DipoleAngle_trajectory(self, ag_single_frame):
        """Test dipole angle module on a single frame."""
        dip = DipoleAngle(ag_single_frame).run()
        assert_allclose(dip.results.cos_theta_i, -0.0821, rtol=1e-3)

    def test_DipoleAngle_trajectory_save(self, ag_single_frame, monkeypatch, tmp_path):
        """
        Test dipole angle module on a single frame.

        Save the result in a text file, and assert that the
        results printed in the file is correct.
        """
        monkeypatch.chdir(tmp_path)

        dipa = DipoleAngle(ag_single_frame).run()
        dipa.save()
        saved_data = np.loadtxt("dipangle.dat")
        assert_allclose(saved_data[1], -0.0821, rtol=1e-3)

    @pytest.mark.parametrize("angle", (0, 30, 60, 90, 180, 272.15))
    def test_orientation_single_molecule_cos(self, angle):
        """Test DipoleAngle module on a single molecule.

        Create a universe with one single water molecule with a given orientation
        'angle' (in degree).

        The expected result is cos(angle).
        """
        ag = line_of_water_molecules(angle_deg=angle)
        assert_allclose(
            DipoleAngle(ag).run().results.cos_theta_i,
            np.cos(np.radians(angle)),
            rtol=1e-3,
            atol=1e-16,
        )

    @pytest.mark.parametrize("angle", (0, 30, 60, 90, 180, 272.15))
    def test_orientation_single_molecule_cos2(self, angle):
        """Test DipoleAngle module on a single molecule.

        Create a universe with one single water molecule with a given orientation
        'angle' (in degree).

        The expected result is cos(angle)**2.
        """
        ag = line_of_water_molecules(angle_deg=angle)
        assert_allclose(
            DipoleAngle(ag).run().results.cos_theta_ii,
            np.cos(np.radians(angle)) ** 2,
            rtol=1e-3,
            atol=1e-16,
        )
