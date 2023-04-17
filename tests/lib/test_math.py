#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test for lib."""
import os
import sys

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

import maicos.lib.math


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data import SPCE_GRO, SPCE_ITP, WATER_GRO, WATER_TPR  # noqa: E402


class Test_sfactor(object):
    """Tests for the sfactor."""

    @pytest.fixture()
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR, WATER_GRO)
        return u.atoms

    @pytest.fixture()
    def qS(self):
        """Define q and S."""
        q = np.array(
            [
                0.25,
                0.25,
                0.25,
                0.36,
                0.36,
                0.36,
                0.44,
                0.51,
                0.51,
                0.51,
                0.56,
                0.56,
                0.56,
                0.56,
                0.56,
                0.56,
                0.62,
                0.62,
                0.62,
                0.71,
                0.71,
                0.71,
                0.76,
                0.76,
                0.76,
                0.76,
                0.76,
                0.76,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.8,
                0.84,
                0.84,
                0.84,
                0.88,
                0.91,
                0.91,
                0.91,
                0.91,
                0.91,
                0.91,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
                0.95,
            ]
        )

        S = np.array(
            [
                1.86430e02,
                6.91000e00,
                8.35300e02,
                5.06760e02,
                1.92540e02,
                1.57790e02,
                9.96500e01,
                5.87470e02,
                7.88630e02,
                5.18170e02,
                4.58650e02,
                1.69000e00,
                3.99910e02,
                6.10340e02,
                1.21359e03,
                4.11800e01,
                9.31980e02,
                6.29120e02,
                9.88500e01,
                3.15220e02,
                1.00840e02,
                1.19420e02,
                2.13180e02,
                4.61770e02,
                3.99640e02,
                8.03880e02,
                1.74830e02,
                3.20900e01,
                1.99190e02,
                4.24690e02,
                1.73552e03,
                1.37732e03,
                1.25050e02,
                2.61750e02,
                4.29610e02,
                2.09000e01,
                2.71450e02,
                4.22340e02,
                1.07590e02,
                3.79520e02,
                6.69000e00,
                5.35330e02,
                1.09210e02,
                6.69970e02,
                1.25354e03,
                3.94200e02,
                1.96100e02,
                1.39890e02,
                8.79600e01,
                4.17020e02,
            ]
        )

        return q, S

    @pytest.mark.parametrize("startq", (0, 0.05))
    @pytest.mark.parametrize("endq", (0.075, 0.1))
    def test_sfactor(self, ag, qS, startq, endq):
        """Test sfactor."""
        q, S = maicos.lib.math.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            startq,
            endq,
            0,  # mintheta
            np.pi,
        )  # maxtheta

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
        """Test sfactor angle."""
        q, S = maicos.lib.math.compute_structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            0,  # startq
            0.5,  # endq
            np.pi / 4,  # mintheta
            np.pi / 2,
        )  # maxtheta

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        assert_almost_equal(
            q, np.array([0.25, 0.25, 0.36, 0.36, 0.36, 0.44]), decimal=2
        )
        assert_almost_equal(
            S, np.array([6.91, 835.3, 192.54, 157.79, 506.76, 99.65]), decimal=2
        )


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

    L is the size of the periodic box. This method should only be used for testing
    against code where one does not want or is not able to use the MDanalysis methods
    (i.e. 1D distances).
    """
    a, b, L = np.array(a), np.array(b), np.array(L)

    return np.linalg.norm((a - b) - np.rint((a - b) / L) * L)


def test_FT():
    """Tests for the Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.lib.math.FT(x, sin)
    assert_almost_equal(abs(t[np.argmax(sin_FT)]), 5, decimal=2)


def test_iFT():
    """Tests for the inverse Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.lib.math.FT(x, sin)
    x_new, sin_new = maicos.lib.math.iFT(t, sin_FT)
    assert_almost_equal(sin, sin_new.real, decimal=1)


def test_symmetrize_even():
    """Tests symmetrization for even array."""
    A_sym = maicos.lib.math.symmetrize(np.arange(10).astype(float))
    assert np.all(A_sym == 4.5)


def test_symmetrize_odd():
    """Tests symmetrization for odd array."""
    A_sym = maicos.lib.math.symmetrize(np.arange(11).astype(float))
    assert np.all(A_sym == 5)


def test_higher_dimensions_length_1():
    """Tests arrays with higher dimensions of length 1."""
    A = np.arange(11).astype(float)[:, np.newaxis]
    A_sym = maicos.lib.math.symmetrize(A)
    A_sym_ref = 5 * np.ones((11, 1))
    assert_equal(A_sym, A_sym_ref)


def test_higher_dimensions():
    """Tests array with higher dimensions."""
    A = np.arange(20).astype(float).reshape(2, 10).T
    A_sym = maicos.lib.math.symmetrize(A)
    assert_equal(A_sym, 9.5)


def test_higher_dimensions_axis():
    """Tests array with higher dimensions with respect to given axis."""
    A = np.arange(20).astype(float).reshape(2, 10).T
    A_sym = maicos.lib.math.symmetrize(A, axis=0)
    A_sym_ref = np.vstack((4.5 * np.ones(10), 14.5 * np.ones(10))).T
    assert_equal(A_sym, A_sym_ref)


def test_symmetrize_inplace():
    """Tests inplace symmetrization."""
    arr = np.arange(11).astype(float)
    maicos.lib.math.symmetrize(arr, inplace=True)
    assert np.all(arr == 5)


@pytest.mark.parametrize(
    ("vector1, vector2, subtract_mean, result"),
    (
        (
            np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
            None,
            False,
            2184.21,
        ),
        (
            np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
            np.vstack((np.linspace(10, 30, 20), np.linspace(30, 50, 20))),
            False,
            5868.42,
        ),
        (
            np.vstack((np.linspace(0, 10, 20), np.linspace(10, 20, 20))),
            np.vstack((np.linspace(10, 30, 20), np.linspace(30, 50, 20))),
            True,
            0.0,
        ),
    ),
)
def test_scalarprod(vector1, vector2, subtract_mean, result):
    """Tests for scalar product."""
    utils_run = maicos.lib.math.scalar_prod_corr(vector1, vector2, subtract_mean)
    assert_almost_equal(np.mean(utils_run), result, decimal=2)


@pytest.mark.parametrize(
    ("vector1, vector2, subtract_mean, result"),
    (
        (np.linspace(0, 20, 50), None, False, 78.23),
        (
            np.linspace(0, 20, 50),
            np.linspace(0, 20, 50) * np.linspace(0, 20, 50),
            False,
            1294.73,
        ),
        (np.linspace(0, 20, 50), None, True, -21.76),
    ),
)
def test_corr(vector1, vector2, subtract_mean, result):
    """Tests for correlation."""
    utils_run = maicos.lib.math.correlation(vector1, vector2, subtract_mean)
    assert_almost_equal(np.mean(utils_run), result, decimal=2)


@pytest.mark.parametrize(
    ("vector1, vector2, subtract_mean, result"),
    (
        (
            2 * generate_correlated_data(int(1e7), 5) + 2,
            None,
            True,
            np.mean(4 * (1 - np.arange(0, 6) / 5)),
        ),
        (
            2 * generate_correlated_data(int(1e7), 5) + 2,
            None,
            False,
            np.mean(4 * (1 - np.arange(0, 6) / 5) + 4),
        ),
    ),
)
def test_corr2(vector1, vector2, subtract_mean, result):
    """Tests for correlation function."""
    utils_run = np.mean(
        maicos.lib.math.correlation(vector1, vector2, subtract_mean)[:6]
    )
    assert_almost_equal(utils_run, result, decimal=2)


@pytest.mark.parametrize(
    ("vector, method, c, mintime, result"),
    (
        (
            generate_correlated_data(int(1e6), 5),
            "sokal",
            8,
            3,
            np.sum(1 - np.arange(1, 5) / 5),
        ),
        (
            generate_correlated_data(int(1e6), 10),
            "sokal",
            8,
            3,
            np.sum(1 - np.arange(1, 10) / 10),
        ),
        (
            generate_correlated_data(int(1e6), 5),
            "chodera",
            8,
            3,
            np.sum(1 - np.arange(1, 5) / 5),
        ),
        (
            generate_correlated_data(int(1e6), 10),
            "chodera",
            8,
            3,
            np.sum(1 - np.arange(1, 10) / 10),
        ),
    ),
)
def test_correlation_time(vector, method, c, mintime, result):
    """Tests for correlation_time."""
    utils_run = maicos.lib.math.correlation_time(vector, method, c, mintime)
    assert_almost_equal(np.mean(utils_run), result, decimal=1)


def test_correlation_time_wrong_method():
    """Tests for correlation_time with wrong method."""
    with pytest.raises(ValueError):
        maicos.lib.math.correlation_time(
            generate_correlated_data(int(1e3), 5), "wrong", 8, 3
        )


def test_new_mean():
    """Tests the new_mean method with random data."""
    series = np.random.rand(100)
    mean = series[0]
    i = 1
    for value in series[1:]:
        i += 1
        mean = maicos.lib.math.new_mean(mean, value, i)
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
        mean = maicos.lib.math.new_mean(mean, value, i)
        var = maicos.lib.math.new_variance(var, old_mean, mean, value, i)
    assert_almost_equal(var, np.std(series) ** 2, decimal=6)


@pytest.mark.parametrize("dim", (0, 1, 2))
@pytest.mark.parametrize("weight", ("mass", "none"))
def test_center_cluster(dim, weight):
    """Tests for pbc com."""
    e_z = np.isin([0, 1, 2], dim)

    dimensions = [20, 30, 100, 90, 90, 90]

    water1 = mda.Universe(SPCE_ITP, SPCE_GRO, topology_format="itp")
    if weight == "mass":
        water1.atoms.translate(-water1.atoms.center_of_mass())
    elif weight == "none":
        water1.atoms.translate(-water1.atoms.center_of_geometry())

    water2 = water1.copy()

    water1.atoms.translate(e_z * dimensions[dim] * 0.2)
    water2.atoms.translate(e_z * dimensions[dim] * 0.8)

    water = mda.Merge(water1.atoms, water2.atoms)
    water.dimensions = dimensions

    if weight == "mass":
        ref_weight = water.atoms.masses
    elif weight == "none":
        ref_weight = np.ones_like(water.atoms.masses)

    for z in np.linspace(0, dimensions[dim], 10):
        water_shifted = water.copy()
        water_shifted.atoms.translate(e_z * z)
        water_shifted.atoms.wrap()
        com = maicos.lib.math.center_cluster(water_shifted.atoms, ref_weight)[dim]
        assert_almost_equal(
            minimum_image_distance(com, z, dimensions[dim]), 0, decimal=5
        )


@pytest.mark.parametrize(
    "vec1, vec2, box, length",
    [
        ([0, 0, 0], [1, 1, 1], [10, 10, 10], np.sqrt(3)),
        ([0, 0, 0], [9, 9, 9], [10, 10, 10], np.sqrt(3)),
        ([0, 0, 0], [9, 19, 29], [10, 20, 30], np.sqrt(3)),
    ],
)
def test_minimal_image(vec1, vec2, box, length):
    """Tests the minimal image function used in other tests."""
    assert minimum_image_distance(vec1, vec2, box) == length
