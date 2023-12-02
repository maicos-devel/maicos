#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderPlanar class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import pytest
from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO, WATER_TPR
from numpy.testing import assert_allclose

from maicos import DiporderSphere


sys.path.append(str(Path(__file__).parents[1]))


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms


class TestDiporderSphere(ReferenceAtomGroups):
    """Tests for the DiporderSphere class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary for test_DiporderSphere_trajectory."""
        res = {}

        res["P0"] = [-0.00174, 0.00057, 0.000126]
        res["cos_theta"] = [-0.107, 0.0331, 0.00747]
        res["cos_2_theta"] = [0.240, 0.308, 0.344]

        return res

    @pytest.mark.parametrize("order_parameter", ["P0", "cos_theta", "cos_2_theta"])
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
