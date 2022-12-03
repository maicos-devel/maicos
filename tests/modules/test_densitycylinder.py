#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DensityCylinder class."""
import os
import sys

import MDAnalysis as mda
import pytest
from numpy.testing import assert_allclose

from maicos import DensityCylinder


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_TPR, WATER_TRR  # noqa: E402


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
        """Test the density profile."""
        dens = DensityCylinder(ag, dens=dens_type).run()
        assert_allclose(dens.results.profile_mean.mean(), mean,
                        atol=1e-4, rtol=1e-2)
