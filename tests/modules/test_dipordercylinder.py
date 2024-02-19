#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderPlanar class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import pytest
from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO_NPT, WATER_TPR_NPT
from numpy.testing import assert_allclose

from maicos import DiporderCylinder


sys.path.append(str(Path(__file__).parents[1]))


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe, single frame."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms


class TestDiporderPlanar(ReferenceAtomGroups):
    """Tests for the DiporderCylinder class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary for test_Diporder_trajectory."""
        res = {}

        # r-direction
        res["r"] = {}
        res["r"]["P0"] = [4.746e-04, -2.40e-04, 7.852e-05]
        res["r"]["cos_theta"] = [2.706e-2, -1.4574e-2, 4.581e-3]

        # z-direction
        res["z"] = {}
        res["z"]["P0"] = [-8.37e-4, -3.84e-4, -2.145e-3]
        res["z"]["cos_theta"] = [-4.922e-2, -2.336e-2, -1.333e-1]

        return res

    @pytest.mark.parametrize("order_parameter", ["P0", "cos_theta"])
    @pytest.mark.parametrize("pdim", ["r", "z"])
    def test_DiporderPlanar_trajectory(
        self, ag_single_frame, pdim, order_parameter, result_dict
    ):
        """Regression test for DiporderCylinder in r and z direction."""
        dip = DiporderCylinder(
            ag_single_frame,
            bin_width=5,
            dim=2,
            pdim=pdim,
            refgroup=ag_single_frame,
            order_parameter=order_parameter,
        ).run()
        assert_allclose(
            dip.results.profile.flatten(), result_dict[pdim][order_parameter], rtol=1e-2
        )
