#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DensitySphere class."""
import sys
from pathlib import Path

import MDAnalysis as mda
import pytest
from numpy.testing import assert_allclose

from maicos import DensitySphere


sys.path.append(str(Path(__file__).parents[1]))
from data import WATER_TPR_NPT, WATER_TRR_NPT  # noqa: E402


class TestDensitySphere(object):
    """Tests for the DensitySphere class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
        return u.atoms

    @pytest.mark.parametrize(
        "dens_type, mean", (("mass", 0.555), ("number", 0.093), ("charge", 2e-4))
    )
    def test_dens(self, ag, dens_type, mean):
        """Test density."""
        dens = DensitySphere(ag, dens=dens_type).run()
        assert_allclose(dens.results.profile.mean(), mean, atol=1e-4, rtol=1e-2)
