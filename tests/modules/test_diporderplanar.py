#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderPlanar class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO_NPT, WATER_TPR_NPT
from numpy.testing import assert_allclose

from maicos import DiporderPlanar

sys.path.append(str(Path(__file__).parents[1]))
from util import line_of_water_molecules  # noqa: E402


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms


class TestDiporderPlanar(ReferenceAtomGroups):
    """Tests for the DiporderPlanar class."""

    @pytest.fixture
    def result_dict(self):
        """Results dictionary for test_Diporder_trajectory."""
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P1"] = [4.253e-03, -5.495e-02, -4.834e-02, -3.119e-02, -5.812e-02]
        res[0]["P2"] = [8.465e-02, 1.079e-02, 1.147e-01, -2.607e-02, -5.956e-02]

        # y-direction
        res[1] = {}
        res[1]["P1"] = [2.244e-02, 4.336e-02, 5.645e-02, 2.108e-02, 4.025e-02]
        res[1]["P2"] = [1.08e-01, 3.377e-02, -9.410e-02, 3.568e-02, -4.201e-03]

        # z-direction
        res[2] = {}
        res[2]["P1"] = [-6.956e-02, -9.645e-02, -7.411e-02, -8.756e-02, -8.259e-02]
        res[2]["P2"] = [-9.168e-02, 1.733e-02, 1.182e-02, -3.268e-02, -1.067e-01]
        return res

    @pytest.mark.parametrize("order_parameter", ["P1", "P2"])
    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_DiporderPlanar_trajectory(
        self, ag_single_frame, dim, order_parameter, result_dict
    ):
        """Regression test for DiporderPlanar in x,y,z direction."""
        dip = DiporderPlanar(
            ag_single_frame,
            bin_width=5,
            dim=dim,
            pdim=dim,
            refgroup=ag_single_frame,
            order_parameter=order_parameter,
        ).run()
        assert_allclose(
            dip.results.profile.flatten(), result_dict[dim][order_parameter], rtol=1e-2
        )

    @pytest.mark.parametrize(("order_parameter", "output"), [("P1", 1), ("P2", 1)])
    def test_DiporderPlanar_3_water_0(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 0."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=0.0)
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(np.mean(dip.results.profile.flatten()), output, atol=1e-3)

    @pytest.mark.parametrize(("order_parameter", "output"), [("P1", 0), ("P2", -0.5)])
    def test_DiporderPlanar_3_water_90(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 90."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=90.0)
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(dip.results.profile.mean(), output, atol=1e-6)

    @pytest.mark.parametrize(
        ("order_parameter", "output"), [("P1", 1 / np.sqrt(2)), ("P2", 0.25)]
    )
    def test_DiporderPlanar_3_water_45(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 45."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=45.0)
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(dip.results.profile.mean(), output, atol=1e-6)

    @pytest.mark.parametrize(
        ("order_parameter", "output"), [("P1", (1 + np.sqrt(2) / 2) / 3), ("P2", 0.25)]
    )
    def test_DiporderPlanar_3_water_mixed_angles(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angles (0, 45, 90)."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=[0.0, 45.0, 90.0])
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(dip.results.profile.mean(), output, atol=1e-6)
