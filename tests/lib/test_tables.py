#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test for tables."""

import numpy as np
from numpy.testing import assert_equal

from maicos.lib.tables import CM_parameters, elements


def test_cm_parameters():
    """Test that the carbon Cromer-Mann parameters are the same as literature value.

    Reference values for carbon are taken from Table 6.1.1.4 in
    https://it.iucr.org/Cb/ch6o1v0001/
    """
    params = CM_parameters["C"]
    assert_equal(params.a, np.array([2.31, 1.02, 1.5886, 0.865]))
    assert_equal(params.b, np.array([20.8439, 10.2075, 0.5687, 51.6512]))
    assert params.c == 0.2156


def test_cm_dummy():
    """Test that DUMMY element has all zero CM parameters."""
    params = CM_parameters["Dummy"]
    assert_equal(params.a, np.zeros(4))
    assert_equal(params.b, np.zeros(4))
    assert params.c == 0.0


def test_elements():
    """Test that the elements in CM_parameters match the known elements."""
    assert set(elements) == set(CM_parameters.keys())
