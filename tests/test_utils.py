#!/usr/bin/env python3
# coding: utf8

import maicos.utils
import numpy as np
from numpy.testing import assert_almost_equal


def test_FT():
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = mdtools.utils.FT(x, sin)
    assert_almost_equal(abs(t[np.argmax(sin_FT)]), 5, decimal=2)


def test_iFT():
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.utils.FT(x, sin)
    x_new, sin_new = maicos.utils.iFT(t, sin_FT)
    assert_almost_equal(sin, sin_new.real, decimal=1)
