#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2020 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np
from numpy.testing import assert_almost_equal
from MDAnalysisTests.core.util import UnWrapUniverse

import maicos.lib.fourier


def test_ft():
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.lib.fourier.ft(x, sin)
    assert_almost_equal(abs(t[np.argmax(sin_FT)]), 5, decimal=2)


def test_ift():
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.lib.fourier.ft(x, sin)
    x_new, sin_new = maicos.lib.fourier.ift(t, sin_FT)
    assert_almost_equal(sin, sin_new.real, decimal=1)
