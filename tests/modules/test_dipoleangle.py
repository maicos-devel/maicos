#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DipoleAngle class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from maicos import DipoleAngle


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_TPR, WATER_TRR  # noqa: E402


class TestDipoleAngle(object):
    """Tests for the DipoleAngle class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.select_atoms('type OW or type H')

    def test_DipoleAngle(self, ag, tmpdir):
        """Test dipole angle."""
        with tmpdir.as_cwd():
            dipa = DipoleAngle(ag)
            dipa.run()
            dipa.save()
            assert_equal(os.path.exists("dipangle.dat"), True)
            assert_almost_equal(np.sum(dipa.cos_theta_i), -3.99, decimal=2)
