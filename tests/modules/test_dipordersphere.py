#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderSphere class."""

import sys
from pathlib import Path

import MDAnalysis as mda
import pytest
from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO_NPT, WATER_TPR_NPT
from numpy.testing import assert_allclose

from maicos import DiporderSphere

sys.path.append(str(Path(__file__).parents[1]))


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


class TestDiporderSphere(ReferenceAtomGroups):
    """Tests for the DiporderSphere class."""

    @pytest.fixture
    def result_dict(self):
        """Results dictionary for test_DiporderSphere_trajectory."""
        res = {}

        res["P1"] = [-0.1074, 0.0331, 0.007478]
        res["P2"] = [-0.1387, -0.03689, 0.01605]

        return res

    @pytest.mark.parametrize("order_parameter", ["P1", "P2"])
    def test_DiporderSphere_trajectory(
        self, ag_single_frame, order_parameter, result_dict
    ):
        """Regression test for DiporderSphere."""
        dip = DiporderSphere(
            ag_single_frame,
            bin_width=5,
            refgroup=ag_single_frame,
            order_parameter=order_parameter,
        ).run()
        assert_allclose(
            dip.results.profile.flatten(), result_dict[order_parameter], rtol=1e-2
        )
