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
import numpy as np
import pytest
from data import AIRWATER_TPR, AIRWATER_TRR, WATER_GRO_NPT, WATER_TPR_NPT
from numpy.testing import assert_allclose

from maicos import DiporderPlanar


sys.path.append(str(Path(__file__).parents[1]))
from util import line_of_water_molecules  # noqa: E402


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
    """Tests for the DiporderPlanar class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary for test_Diporder_trajectory."""
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P0"] = [6.33e-05, -9.26e-04, -8.31e-04, -4.92e-04, -9.56e-04]
        res[0]["cos_theta"] = [4.25e-3, -5.49e-2, -4.83e-2, -3.1188e-2, -5.8e-2]
        res[0]["cos_2_theta"] = [3.90e-1, 3.41e-1, 4.10e-1, 3.16e-1, 2.94e-1]

        # y-direction
        res[1] = {}
        res[1]["P0"] = [0.000344, 0.000734, 0.00092, 0.000335, 0.000667]
        res[1]["cos_theta"] = [0.022442, 0.043359, 0.056447, 0.021081, 0.040248]
        res[1]["cos_2_theta"] = [4.05e-1, 3.56e-1, 2.71e-1, 3.57e-1, 3.31e-1]

        # z-direction
        res[2] = {}
        res[2]["P0"] = [-1.12e-3, -1.61e-3, -1.22e-3, -1.44e-3, -1.29e-3]
        res[2]["cos_theta"] = [-6.96e-2, -9.65e-2, -7.41e-2, -8.76e-2, -8.26e-2]
        res[2]["cos_2_theta"] = [2.72e-1, 3.45e-1, 3.41e-1, 3.12e-1, 2.62e-1]
        return res

    @pytest.mark.parametrize("order_parameter", ["P0", "cos_theta", "cos_2_theta"])
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

    @pytest.mark.parametrize(
        "order_parameter, output", [("P0", 0), ("cos_theta", 1), ("cos_2_theta", 1)]
    )
    def test_DiporderPlanar_3_water_0(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 0."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=0.0)
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(np.mean(dip.results.profile.flatten()), output, atol=1e-3)

    @pytest.mark.parametrize(
        "order_parameter, output", [("P0", 0), ("cos_theta", 0), ("cos_2_theta", 0)]
    )
    def test_DiporderPlanar_3_water_90(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 90."""
        ag = line_of_water_molecules(n_molecules=3, angle_deg=90.0)
        dip = DiporderPlanar(ag, bin_width=10, order_parameter=order_parameter).run()
        assert_allclose(dip.results.profile.mean(), output, atol=1e-6)
