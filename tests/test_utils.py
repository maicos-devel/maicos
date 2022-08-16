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
from modules.datafiles import LAMMPS10WATER, SPCE_GRO, SPCE_ITP
from numpy.testing import assert_almost_equal, assert_equal

import maicos.utils


def generate_correlated_data(T, repeat, seed=0):
    """Generate correlated data to be used in test_correlation_time.

    T : int
        length of timeseries to be generated
    corr_t : int
        correlation time in step size
    seed : int
        seed the random number generator
    returns : ndarray, shape (n,)
    """
    if seed is not None:
        np.random.seed(seed)

    t = T // repeat
    return np.repeat(np.random.normal(size=t), repeat)


def minimum_image_distance(a, b, L):
    """Return the minimum image distance of two vectors.

    L is the size of the periodic box. This method should only be
    used for testing against code where one does not want or is
    not able to use the MDanalysis methods (i.e. 1D distances).
    """
    a, b, L = np.array(a), np.array(b), np.array(L)

    return np.linalg.norm((a - b) - np.rint((a - b) / L) * L)


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
    A_sym = maicos.utils.symmetrize(np.arange(10).astype(float))
    assert np.all(A_sym == 4.5)


def test_symmetrize_odd():
    """Tests symmetrization for odd array."""
    A_sym = maicos.utils.symmetrize(np.arange(11).astype(float))
    assert np.all(A_sym == 5)


def test_higher_dimensions_length_1():
    """Tests arrays with higher dimensions of length 1."""
    A = np.arange(11).astype(float)[:, np.newaxis]
    A_sym = maicos.utils.symmetrize(A)
    A_sym_ref = 5 * np.ones((11, 1))
    assert_equal(A_sym, A_sym_ref)


def test_higher_dimensions():
    """Tests array with higher dimensions."""
    A = np.arange(20).astype(float).reshape(2, 10).T
    A_sym = maicos.utils.symmetrize(A)
    assert_equal(A_sym, 9.5)


def test_higher_dimensions_axis():
    """Tests array with higher dimensions with respect to given axis."""
    A = np.arange(20).astype(float).reshape(2, 10).T
    A_sym = maicos.utils.symmetrize(A, axis=0)
    A_sym_ref = np.vstack((4.5 * np.ones(10), 14.5 * np.ones(10))).T
    assert_equal(A_sym, A_sym_ref)


def test_symmetrize_inplace():
    """Tests inplace symmetrization."""
    arr = np.arange(11).astype(float)
    maicos.utils.symmetrize(arr, inplace=True)
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
        assert maicos.utils.get_cli_input() == 'maicos foo "foo bar"'


@pytest.mark.parametrize(
    ('vector1, vector2, subtract_mean, result'),
    (
        (np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
         None, False, 2184.21),
        (np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
         np.vstack((np.linspace(10, 30, 20), np.linspace(30, 50, 20))),
         False, 5868.42),
        (np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
         np.vstack((np.linspace(10, 30, 20), np.linspace(30, 50, 20))),
         True, 0.0),

        ),
    )
def test_scalarprod(vector1, vector2, subtract_mean, result):
    """Tests for scalar product."""
    utils_run = maicos.utils.ScalarProdCorr(vector1, vector2, subtract_mean)
    assert_almost_equal(np.mean(utils_run), result, decimal=2)


@pytest.mark.parametrize(
    ('vector1, vector2, subtract_mean, result'),
    (
        (np.linspace(0, 20, 50), None, False, 78.23),
        (np.linspace(0, 20, 50), np.linspace(0, 20, 50)
         * np.linspace(0, 20, 50), False, 1294.73),
        (np.linspace(0, 20, 50), None, True, -21.76),

        ),
    )
def test_corr(vector1, vector2, subtract_mean, result):
    """Tests for correlation."""
    utils_run = maicos.utils.Correlation(vector1, vector2, subtract_mean)
    assert_almost_equal(np.mean(utils_run), result, decimal=2)


@pytest.mark.parametrize(
    ('vector1, vector2, subtract_mean, result'),
    (
        (2 * generate_correlated_data(int(1E7), 5) + 2,
         None, True, np.mean(4 * (1 - np.arange(0, 6) / 5))),
        (2 * generate_correlated_data(int(1E7), 5) + 2,
         None, False, np.mean(4 * (1 - np.arange(0, 6) / 5) + 4)),
        ),
    )
def test_corr2(vector1, vector2, subtract_mean, result):
    """Tests for Correlation function."""
    utils_run = np.mean(maicos.utils.Correlation(vector1, vector2,
                                                 subtract_mean)[:6])
    assert_almost_equal(utils_run, result, decimal=2)


@pytest.mark.parametrize(
    ('vector, method, c, mintime, result'),
    (
        (generate_correlated_data(int(1E6), 5), 'Sokal', 8, 3,
         np.sum(1 - np.arange(1, 5) / 5)),
        (generate_correlated_data(int(1E6), 10), 'Sokal', 8, 3,
         np.sum(1 - np.arange(1, 10) / 10)),
        (generate_correlated_data(int(1E6), 5), 'Chodera', 8, 3,
         np.sum(1 - np.arange(1, 5) / 5)),
        (generate_correlated_data(int(1E6), 10), 'Chodera', 8, 3,
         np.sum(1 - np.arange(1, 10) / 10)),
        ),
    )
def test_correlation_time(vector, method, c, mintime, result):
    """Tests for correlation_time."""
    utils_run = maicos.utils.correlation_time(vector, method, c, mintime)
    assert_almost_equal(np.mean(utils_run), result, decimal=1)


def test_new_mean():
    """Tests the new_mean method with random data."""
    series = np.random.rand(100)
    mean = series[0]
    i = 1
    for value in series[1:]:
        i += 1
        mean = maicos.utils.new_mean(mean, value, i)
    assert_almost_equal(mean, np.mean(series), decimal=6)


def test_new_variance():
    """Tests the new_variance method with random data."""
    series = np.random.rand(100)
    var = 0
    mean = series[0]
    i = 1
    for value in series[1:]:
        i += 1
        old_mean = mean
        mean = maicos.utils.new_mean(mean, value, i)
        var = maicos.utils.new_variance(var, old_mean, mean, value, i)
    assert_almost_equal(var, np.std(series)**2, decimal=6)


@pytest.mark.parametrize('dim', (0, 1, 2))
def test_cluster_com(dim):
    """Tests for pbc com."""
    e_z = np.isin([0, 1, 2], dim)

    dimensions = [20, 30, 100, 90, 90, 90]

    water1 = mda.Universe(SPCE_ITP, SPCE_GRO, topology_format='itp')
    water1.atoms.translate(-water1.atoms.center_of_mass())

    water2 = water1.copy()

    water1.atoms.translate(e_z * dimensions[dim] * 0.2)
    water2.atoms.translate(e_z * dimensions[dim] * 0.8)

    water = mda.Merge(water1.atoms, water2.atoms)
    water.dimensions = dimensions

    for z in np.linspace(0, dimensions[dim], 10):
        water_shifted = water.copy()
        water_shifted.atoms.translate(e_z * z)
        water_shifted.atoms.wrap()
        com = maicos.utils.cluster_com(water_shifted.atoms)[dim]
        assert_almost_equal(minimum_image_distance(com, z, dimensions[dim]),
                            0, decimal=5)


@pytest.mark.parametrize('vec1, vec2, box, length',
                         [([0, 0, 0], [1, 1, 1], [10, 10, 10], np.sqrt(3)),
                          ([0, 0, 0], [9, 9, 9], [10, 10, 10], np.sqrt(3)),
                          ([0, 0, 0], [9, 19, 29], [10, 20, 30], np.sqrt(3))])
def test_minimal_image(vec1, vec2, box, length):
    """Tests the minimal image function used in other tests."""
    assert minimum_image_distance(vec1, vec2, box) == length
