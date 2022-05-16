#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the utilities."""

import sys
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysisTests.core.util import UnWrapUniverse
from modules.datafiles import LAMMPS10WATER
from numpy.testing import assert_almost_equal, assert_equal

import maicos.utils


def test_FT():
    """Tests for the Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.utils.FT(x, sin)
    assert_almost_equal(abs(t[np.argmax(sin_FT)]), 5, decimal=2)


def test_iFT():
    """Tests for the inverse Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.utils.FT(x, sin)
    x_new, sin_new = maicos.utils.iFT(t, sin_FT)
    assert_almost_equal(sin, sin_new.real, decimal=1)


def test_symmetrize_even():
    """Tests symmetrization for even array."""
    sym_arr = maicos.utils.symmetrize_1D(np.arange(10).astype(float))
    assert np.all(sym_arr == 4.5)


def test_symmetrize_odd():
    """Tests symmetrization for odd array."""
    sym_arr = maicos.utils.symmetrize_1D(np.arange(11).astype(float))
    assert np.all(sym_arr == 5)


def test_higher_dimensions():
    """Tests arrays with higher dimensions of length 1."""
    arr = np.arange(11).astype(float)[:, np.newaxis]
    sym_arr = maicos.utils.symmetrize_1D(arr)
    sym_arr_ref = 5 * np.ones((11, 1))
    assert_equal(sym_arr, sym_arr_ref)


@pytest.mark.parametrize("shape", [(2, 2), (1, 11, 1)])
def test_not_allowed_dimensions(shape):
    """Tests error raise for higher dimensions."""
    with pytest.raises(ValueError, match="Only 1 dimensional arrays"):
        maicos.utils.symmetrize_1D(np.ones(shape))


def test_symmetrize_inplace():
    """Tests inplace symmetrization."""
    arr = np.arange(11).astype(float)
    maicos.utils.symmetrize_1D(arr, inplace=True)
    assert np.all(arr == 5)


def test_check_compound():
    """Tests check compound."""
    u = UnWrapUniverse()
    assert maicos.utils.check_compound(u.atoms) == "molecules"

    u = UnWrapUniverse(have_molnums=False, have_bonds=True)
    assert maicos.utils.check_compound(u.atoms) == "fragments"

    u = UnWrapUniverse(have_molnums=False, have_bonds=False)
    assert maicos.utils.check_compound(u.atoms) == "residues"


def test_sort_atomsgroup_lammps():
    """Tests sort atoms group LAMMPS."""
    u = mda.Universe(LAMMPS10WATER)
    atoms = maicos.utils.sort_atomgroup(u.atoms)

    assert np.all(np.diff(atoms.fragindices) >= 0)


def test_get_cli_input():
    """Tests get cli input."""
    testargs = ['maicos', 'foo', "foo bar"]
    with patch.object(sys, 'argv', testargs):
        assert maicos.utils.get_cli_input() == 'Command line was: ' \
                                               'maicos foo "foo bar"'
