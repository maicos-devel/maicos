#!/usr/bin/env python3
"""Tests for the timeseries modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import MDAnalysis as mda
import numpy as np
import pytest
from datafiles import NVE_WATER_TPR, NVE_WATER_TRR, WATER_TPR, WATER_TRR
from numpy.testing import assert_almost_equal

from maicos import DipoleAngle, KineticEnergy


class TestDipoleAngle(object):
    """Tests for the DipoleAngle class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.select_atoms('type OW or type H')

    def test_DipoleAngle(self, ag):
        """Test dipole angle."""
        dipa = DipoleAngle(ag)
        dipa.run()
        assert_almost_equal(np.sum(dipa.cos_theta_i), -3.99, decimal=2)


class TestKineticEnergy(object):
    """Tests for the KineticEnergy class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(NVE_WATER_TPR, NVE_WATER_TRR)
        return u.atoms

    def test_ke_trans(self, ag):
        """Test translational kinetic energy."""
        ke = KineticEnergy(ag).run()
        assert_almost_equal(np.mean(ke.results.trans), 2156, decimal=0)

    def test_ke_rot(self, ag):
        """Test rotational kinetic energy."""
        ke = KineticEnergy(ag).run()
        assert_almost_equal(np.mean(ke.results.rot), 2193, decimal=0)
