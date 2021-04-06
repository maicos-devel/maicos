#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2021 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later


import pytest

from numpy.testing import assert_equal, assert_almost_equal

import MDAnalysis as mda
from datafiles import WATER_TPR, WATER_TRR

from maicos.modules.base import _AnalysisBase


class Test_AnalysisBase(object):

    @pytest.fixture()
    def trajectory(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.trajectory

    def test_large_begin(self, trajectory):

        with pytest.raises(ValueError):
            _AnalysisBase(trajectory).run(begin=trajectory.totaltime + 1)

    def test_small_end(self, trajectory):

        with pytest.raises(ValueError):
            _AnalysisBase(trajectory).run(end=trajectory.dt / 10)