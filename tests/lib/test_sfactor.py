#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import MDAnalysis as mda
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from maicos.lib import sfactor

from modules.datafiles import WATER_GRO, WATER_TPR


class Test_sfactor(object):
    @pytest.fixture()
    def ag(self):
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.fixture()
    def qS(self):

        q = np.array([
            0.25, 0.25, 0.25, 0.36, 0.36, 0.36, 0.44, 0.51, 0.51, 0.51, 0.56,
            0.56, 0.56, 0.56, 0.56, 0.56, 0.62, 0.62, 0.62, 0.71, 0.71, 0.71,
            0.76, 0.76, 0.76, 0.76, 0.76, 0.76, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.84, 0.84, 0.84, 0.88, 0.91, 0.91, 0.91, 0.91, 0.91, 0.91, 0.95,
            0.95, 0.95, 0.95, 0.95, 0.95
        ])

        S = np.array([
            1.86430e+02, 6.91000e+00, 8.35300e+02, 5.06760e+02, 1.92540e+02,
            1.57790e+02, 9.96500e+01, 5.87470e+02, 7.88630e+02, 5.18170e+02,
            4.58650e+02, 1.69000e+00, 3.99910e+02, 6.10340e+02, 1.21359e+03,
            4.11800e+01, 9.31980e+02, 6.29120e+02, 9.88500e+01, 3.15220e+02,
            1.00840e+02, 1.19420e+02, 2.13180e+02, 4.61770e+02, 3.99640e+02,
            8.03880e+02, 1.74830e+02, 3.20900e+01, 1.99190e+02, 4.24690e+02,
            1.73552e+03, 1.37732e+03, 1.25050e+02, 2.61750e+02, 4.29610e+02,
            2.09000e+01, 2.71450e+02, 4.22340e+02, 1.07590e+02, 3.79520e+02,
            6.69000e+00, 5.35330e+02, 1.09210e+02, 6.69970e+02, 1.25354e+03,
            3.94200e+02, 1.96100e+02, 1.39890e+02, 8.79600e+01, 4.17020e+02
        ])

        return q, S

    @pytest.mark.parametrize('startq', (0, 0.5))
    @pytest.mark.parametrize('endq', (0.75, 1))
    def test_sfactor(self, ag, qS, startq, endq):
        q, S = sfactor.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            startq,
            endq,
            0,  # mintheta
            np.pi)  # maxtheta

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        # Get indices to slice qS array
        sel_indices = np.logical_and(startq < qS[0], qS[0] < endq)

        assert_almost_equal(q, qS[0][sel_indices], decimal=2)

        # Only check S for full q width
        if startq == 0 and endq == 1:
            assert_almost_equal(S, qS[1], decimal=2)

    def test_sfactor_angle(self, ag):
        q, S = sfactor.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            0,  # startq
            0.5,  # endq
            np.pi / 4,  # mintheta
            np.pi / 2)  # maxtheta

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        assert_almost_equal(q,
                            np.array([0.25, 0.25, 0.36, 0.36, 0.36, 0.44]),
                            decimal=2)
        assert_almost_equal(S,
                            np.array(
                                [6.91, 835.3, 192.54, 157.79, 506.76, 99.65]),
                            decimal=2)
