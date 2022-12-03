#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Saxs class."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from maicos import Saxs


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import WATER_TPR, WATER_TRR  # noqa: E402


class TestSaxs(object):
    """Tests for the Saxs class."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.atoms

    def test_saxs(sef, ag):
        """Test Saxs."""
        Saxs(ag, endq=20).run(stop=5)

    def test_one_frame(self, ag):
        """Test analysis running for one frame.

        Test if the division by the number of frames is correct.
        """
        saxs = Saxs(ag, endq=20).run(stop=1)
        assert not np.isnan(saxs.results.scat_factor).any()

    def test_theta(self, ag, tmpdir):
        """Test min & max theta conditions."""
        with tmpdir.as_cwd():
            saxs = Saxs(ag, mintheta=-10, maxtheta=190)
            saxs.run()
            saxs.save()
            assert_allclose(saxs.mintheta, 0)
            assert_equal(os.path.exists("sq.dat"), True)

    def test_nobindata(self, ag, tmpdir):
        """Test when nobindata is True."""
        with tmpdir.as_cwd():
            saxs = Saxs(ag, nobin=True)
            saxs.run()
            assert_equal(type(saxs.q_factor).__name__ == 'ndarray', True)
            saxs.save()
            assert_equal(os.path.exists("sq.dat"), True)
