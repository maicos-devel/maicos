#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the KineticEnergy class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from maicos import KineticEnergy


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import NVE_WATER_TPR, NVE_WATER_TRR  # noqa: E402


class TestKineticEnergy(object):
    """Tests for the KineticEnergy class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(NVE_WATER_TPR, NVE_WATER_TRR)
        return u.atoms

    def test_ke_trans(self, ag):
        """Test translational kinetic energy."""
        ke = KineticEnergy(ag, refpoint="COM").run()
        assert_almost_equal(np.mean(ke.results.trans), 2156, decimal=0)

    def test_ke_rot(self, ag, tmpdir):
        """Test rotational kinetic energy."""
        with tmpdir.as_cwd():
            ke = KineticEnergy(ag).run()
            ke.save()
            assert_equal(os.path.exists("ke.dat"), True)
            assert_almost_equal(np.mean(ke.results.rot), 2193, decimal=0)

    def test_prepare(self, ag):
        """Test Value error when refpoint is not COM or COC."""
        kem = KineticEnergy(ag, refpoint="OH")
        with assert_raises(ValueError):
            kem.run()

    def test_ke_rot_COC(self, ag):
        """Test rotational KE COC."""
        ke = KineticEnergy(ag, refpoint="COC").run()
        assert_almost_equal(np.mean(ke.results.rot), 746, decimal=0)
