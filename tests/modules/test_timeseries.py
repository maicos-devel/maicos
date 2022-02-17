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

from maicos import DipoleAngle, KineticEnergy
import numpy as np
from numpy.testing import assert_almost_equal

from datafiles import WATER_TPR, WATER_TRR, NVE_WATER_TPR, NVE_WATER_TRR

class TestDipoleAngle(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_TRR)
        return u.select_atoms('type OW or type H')

    def test_DipoleAngle(self, ag):
        dipa = DipoleAngle(ag)
        dipa.run()
        assert_almost_equal(np.sum(dipa.cos_theta_i),-3.99,decimal=2)


class TestKineticEnergy(object):

    @pytest.fixture()
    def ag(self):
        u = mda.Universe(NVE_WATER_TPR, NVE_WATER_TRR)
        return u.atoms

    def test_ke_trans(self, ag):
        ke = KineticEnergy(ag).run()
        assert_almost_equal(np.mean(ke.results.trans), 2156, decimal=0)

    def test_ke_rot(self, ag):
        ke = KineticEnergy(ag).run()
        assert_almost_equal(np.mean(ke.results.rot), 2193, decimal=0)
