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
from MDAnalysisTests.core.util import UnWrapUniverse
from modules.datafiles import LAMMPS10WATER
from numpy.testing import assert_almost_equal

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
