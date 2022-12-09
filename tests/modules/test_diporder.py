#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DiporderPlanar class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from create_mda_universe import line_of_water_molecules
from numpy.testing import assert_allclose

from maicos import DiporderPlanar


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import AIRWATER_TPR, AIRWATER_TRR  # noqa: E402


class TestDiporderPlanar(object):
    """Tests for the DiporderPlanar class."""

    @pytest.fixture()
    def result_dict(self):
        """Results dictionary."""
        res = {}

        # x-direction
        res[0] = {}
        res[0]["P0"] = 4 * [0]
        res[0]["cos_theta"] = 4 * [0]
        res[0]["cos_2_theta"] = 4 * [0.35]

        # y-direction must be the same as x
        res[1] = res[0]

        # z-direction
        res[2] = {}
        res[2]["P0"] = 12 * [0]
        res[2]["cos_theta"] = 2 * [np.nan] + 8 * [0] + 2 * [np.nan]
        res[2]["cos_2_theta"] = [np.nan, np.nan, 0.06, 0.25, 0.33, 0.33,
                                 0.33, 0.33, 0.26, 0.09, np.nan, np.nan]

        return res

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(AIRWATER_TPR, AIRWATER_TRR)
        return u.atoms

    @pytest.mark.parametrize('order_parameter',
                             ['P0', 'cos_theta', 'cos_2_theta'])
    @pytest.mark.parametrize('dim', [0, 1, 2])
    def test_DiporderPlanar_slab(self, ag, dim, order_parameter, result_dict):
        """Test DiporderPlanar for slab system in x,y,z direction."""
        dip = DiporderPlanar(ag,
                             bin_width=5,
                             dim=dim,
                             refgroup=ag,
                             order_parameter=order_parameter).run()
        assert_allclose(dip.results.profile.flatten(),
                        result_dict[dim][order_parameter],
                        atol=1e-1)

    @pytest.mark.parametrize('order_parameter, output',
                             [('P0', 0), ('cos_theta', 1), ('cos_2_theta', 1)])
    def test_DiporderPlanar_3_water_0(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 0."""
        group_H2O_1 = line_of_water_molecules(n_molecules=3, angle_deg=0)
        dip = DiporderPlanar(group_H2O_1, bin_width=10,
                             order_parameter=order_parameter).run()

        assert_allclose(np.mean(dip.results.profile.flatten()),
                        output, atol=1e-3)

    @pytest.mark.parametrize('order_parameter, output',
                             [('P0', 0), ('cos_theta', 0), ('cos_2_theta', 0)])
    def test_DiporderPlanar_3_water_90(self, order_parameter, output):
        """Test DiporderPlanar for 3 water molecules with angle 90."""
        group_H2O_2 = line_of_water_molecules(n_molecules=3, angle_deg=90)
        dip = DiporderPlanar(group_H2O_2, bin_width=10,
                             order_parameter=order_parameter).run()

        assert_allclose(np.mean(dip.results.profile.flatten()),
                        output, atol=1e-6)
