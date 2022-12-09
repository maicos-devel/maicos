#!/usr/bin/env python3
"""Tests for the SAXS modules."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import sys

import MDAnalysis as mda
import pytest
from data import WATER_GRO, WATER_TPR
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from maicos import Saxs


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class ReferenceAtomGroups:
    """Super class with methods reference AtomGroups for tests."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms


class TestSaxs(ReferenceAtomGroups):
    """Tests for the Saxs class."""

    @pytest.fixture()
    def ag_single_frame(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    def test_one_frame(sef, ag_single_frame):
        """
        Test Saxs on one frame.

        Test if the division by the number of frames is correct.
        """
        saxs = Saxs(ag_single_frame, endq=20).run()
        assert_almost_equal(saxs.results.scat_factor[0], 1.6047, decimal=3)

    def test_theta(self, ag_single_frame, tmpdir):
        """Test min & max theta conditions on one frame."""
        with tmpdir.as_cwd():
            saxs = Saxs(ag_single_frame, mintheta=-10, maxtheta=190).run()
            saxs.save()
            assert_allclose(saxs.mintheta, 0)
            assert_equal(os.path.exists("sq.dat"), True)

    def test_nobindata(self, ag_single_frame):
        """Test when nobindata is True."""
        saxs = Saxs(ag_single_frame, nobin=True).run()
        assert_equal(type(saxs.q_factor).__name__ == 'ndarray', True)
