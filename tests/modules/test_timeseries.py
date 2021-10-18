#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import MDAnalysis as mda
import pytest

from maicos import dipole_angle
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import WATER_GRO, WATER_TPR, WATER_TRR

class Test_dipole_angle(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.select_atoms('type OW or type H')

    def test_dipole_angle(self, ag):
        dipa = dipole_angle(ag)
        dipa.run()
        assert_almost_equal(np.sum(dipa.cos_theta_i),-3.99,decimal=2)
