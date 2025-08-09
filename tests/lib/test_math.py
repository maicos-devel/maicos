#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test for lib."""

import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import maicos.lib.math
import maicos.lib.util

sys.path.append(str(Path(__file__).parents[1]))
from data import SPCE_GRO, SPCE_ITP  # noqa: E402


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


def test_symmetrize_even():
    """Tests symmetrization for even array."""
    A_sym = maicos.lib.math.symmetrize(np.arange(10).astype(float))
    assert np.all(A_sym == 4.5)


def test_symmetrize_odd():
    """Tests symmetrization for odd array."""
    A_sym = maicos.lib.math.symmetrize(np.arange(11).astype(float))
    assert np.all(A_sym == 5)


def test_symmetrize_parity_even():
    """Tests symmetrization for even parity."""
    A_sym = maicos.lib.math.symmetrize(np.arange(11).astype(float), is_odd=False)
    assert np.all(A_sym == 5)


def test_symmetrize_parity_odd():
    """Tests symmetrization for odd parity."""
    A = np.arange(10).astype(float)
    A_result = np.arange(10).astype(float) - 4.5
    A_sym = maicos.lib.math.symmetrize(A, is_odd=True)
    assert np.all(A_sym == A_result)


def test_symmetrize_parity_odd_antisymmetric():
    """Tests symmetrization for odd parity.

    The array is unchanged, as it is already antisymmetric.
    """
    A = np.arange(11).astype(float) - 5
    A_sym = maicos.lib.math.symmetrize(A, is_odd=True)
    assert np.all(A_sym == A)


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
    ("vector1", "vector2", "subtract_mean", "result"),
    [
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
    ],
)
def test_scalarprod(vector1, vector2, subtract_mean, result):
    """Tests for scalar product."""
    utils_run = maicos.lib.math.scalar_prod_corr(vector1, vector2, subtract_mean)
    assert_allclose(np.mean(utils_run), result, rtol=1e-2)


@pytest.mark.parametrize(
    ("vector1", "vector2", "subtract_mean", "result"),
    [
        (np.linspace(0, 20, 50), None, False, 78.23),
        (
            np.linspace(0, 20, 50),
            np.linspace(0, 20, 50) * np.linspace(0, 20, 50),
            False,
            1294.73,
        ),
        (np.linspace(0, 20, 50), None, True, -21.76),
    ],
)
def test_corr(vector1, vector2, subtract_mean, result):
    """Tests for correlation."""
    utils_run = maicos.lib.math.correlation(vector1, vector2, subtract_mean)
    assert_allclose(np.mean(utils_run), result, rtol=1e-2)


@pytest.mark.parametrize(
    ("vector1", "vector2", "subtract_mean", "result"),
    [
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
    ],
)
def test_corr2(vector1, vector2, subtract_mean, result):
    """Tests for correlation function."""
    utils_run = np.mean(
        maicos.lib.math.correlation(vector1, vector2, subtract_mean)[:6]
    )
    assert_allclose(utils_run, result, rtol=1e-2)


@pytest.mark.parametrize(
    ("vector", "method", "result"),
    [
        (
            generate_correlated_data(int(1e6), 5),
            "sokal",
            np.sum(1 - np.arange(1, 5) / 5),
        ),
        (
            generate_correlated_data(int(1e6), 10),
            "sokal",
            np.sum(1 - np.arange(1, 10) / 10),
        ),
        (
            generate_correlated_data(int(1e6), 5),
            "chodera",
            np.sum(1 - np.arange(1, 5) / 5),
        ),
        (
            generate_correlated_data(int(1e6), 10),
            "chodera",
            np.sum(1 - np.arange(1, 10) / 10),
        ),
    ],
)
def test_correlation_time(vector, method, result):
    """Tests for correlation_time."""
    utils_run = maicos.lib.math.correlation_time(vector, method)
    assert_allclose(np.mean(utils_run), result, rtol=1e-1)


def test_correlation_time_wrong_method():
    """Tests for correlation_time with wrong method."""
    match = "Unknown method: wrong. Chose either 'sokal' or 'chodera'."
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.correlation_time(
            generate_correlated_data(int(1e3), 5),
            method="wrong",
        )


def test_correlation_large_mintime():
    """Tests for correlation_time with where mintime is larger than timeseries."""
    match = "has to be smaller then the length of `timeseries`"
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.correlation_time(
            generate_correlated_data(int(1e3), 5), mintime=2e3
        )


def test_new_mean():
    """Tests the new_mean method with random data."""
    series = np.random.rand(100)
    mean = series[0]
    i = 1
    for value in series[1:]:
        i += 1
        mean = maicos.lib.math.new_mean(mean, value, i)
    assert_allclose(mean, np.mean(series), rtol=1e-6)


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
    assert_allclose(var, np.std(series) ** 2, rtol=1e-6)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("weight", ["mass", "none"])
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
        assert_allclose(minimum_image_distance(com, z, dimensions[dim]), 0, atol=1e-4)


@pytest.mark.parametrize(
    ("vec1", "vec2", "box", "length"),
    [
        ([0, 0, 0], [1, 1, 1], [10, 10, 10], np.sqrt(3)),
        ([0, 0, 0], [9, 9, 9], [10, 10, 10], np.sqrt(3)),
        ([0, 0, 0], [9, 19, 29], [10, 20, 30], np.sqrt(3)),
    ],
)
def test_minimal_image(vec1, vec2, box, length):
    """Tests the minimal image function used in other tests."""
    assert minimum_image_distance(vec1, vec2, box) == length


def test_transform_sphere():
    """Test spherical transformation of positions."""
    u = mda.Universe.empty(n_atoms=4, trajectory=True)

    # Manipulate universe
    u.dimensions = np.array([2, 2, 2, 90, 90, 90])

    sel = u.atoms[:4]

    # Put one atom at each quadrant on different z positions
    sel[0].position = np.array([0, 0, 1])
    sel[1].position = np.array([0, 2, 1])
    sel[2].position = np.array([2, 2, 1])
    sel[3].position = np.array([2, 0, 1])

    pos_sph = maicos.lib.math.transform_sphere(
        u.atoms.positions, origin=u.dimensions[:3] / 2
    )

    assert_allclose(pos_sph[:, 0], np.sqrt(2))

    # phi component
    assert_allclose(pos_sph[0, 1], np.arctan(1) - np.pi)
    assert_allclose(pos_sph[1, 1], np.arctan(-1) + np.pi)
    assert_allclose(pos_sph[2, 1], np.arctan(1))
    assert_allclose(pos_sph[3, 1], np.arctan(-1))

    # theta component
    assert_allclose(pos_sph[:, 2], np.arccos(0))


def test_transform_cylinder():
    """Test cylinder transformation of positions."""
    u = mda.Universe.empty(4, trajectory=True)

    # Manipulate universe
    u.dimensions = np.array([2, 2, 2, 90, 90, 90])

    sel = u.atoms

    # Put one atom at each quadrant on different z positions
    sel[0].position = np.array([0, 0, 1])
    sel[1].position = np.array([0, 2, 2])
    sel[2].position = np.array([2, 2, 3])
    sel[3].position = np.array([2, 0, 4])

    pos_cyl = maicos.lib.math.transform_cylinder(
        sel.positions, origin=u.dimensions[:3] / 2, dim=2
    )

    # r component
    assert_allclose(pos_cyl[:, 0], np.sqrt(2))

    # phi component
    assert_allclose(pos_cyl[0, 1], np.arctan(1) - np.pi)
    assert_allclose(pos_cyl[1, 1], np.arctan(-1))
    assert_allclose(pos_cyl[2, 1], np.arctan(1))
    assert_allclose(pos_cyl[3, 1], np.arctan(-1) + np.pi)

    # z component
    assert_equal(pos_cyl[:, 2], sel.positions[:, 2])


@pytest.mark.parametrize(
    ("n_A", "n_B"),
    [
        (1, 1),  # trivial case
        (1, 10),  # series B is larger than the other
        (10, 1),  # series A is larger than the other
        (10000, 10000),  # both series are large
        (10000, 1),  # series A is much larger than series B
        (1, 10000),  # series B is much larger than series A
    ],
)
def test_combine_subsample_variance(n_A, n_B):
    """Test parallel Welford algorithm for mean and variance."""
    # Ensure series_B is different from series_A
    series_A = np.random.rand(n_A) * 100 - 50
    series_B = np.random.rand(n_B) * 10
    series_AB = np.concatenate([series_A, series_B])

    n_A = len(series_A)
    n_B = len(series_B)

    mu_A = series_A.mean()
    mu_B = series_B.mean()

    M_A = len(series_A) * series_A.var()
    M_B = len(series_B) * series_B.var()

    n_AB, mu_AB, M_AB = maicos.lib.math.combine_subsample_variance(
        n_A, n_B, mu_A, mu_B, M_A, M_B
    )

    assert n_AB == len(series_AB)
    assert_allclose(mu_AB, series_AB.mean(), rtol=1e-9)
    assert_allclose(M_AB, series_AB.var() * len(series_AB), rtol=1e-9)


def test_combine_subsample_variance_empty():
    """Test parallel Welford algorithm with empty series."""
    series_A = np.random.rand(10)
    n_A = len(series_A)
    n_B = 0  # empty series
    mu_A = series_A.mean()
    mu_B = np.nan  # mean of empty series is NaN
    M_A = len(series_A) * series_A.var()
    M_B = np.nan

    n_AB, mu_AB, M_AB = maicos.lib.math.combine_subsample_variance(
        n_A, n_B, mu_A, mu_B, M_A, M_B
    )

    assert n_AB == len(series_A)
    assert mu_AB == mu_A
    assert M_AB == M_A

    # Test the other way around, series_A is empty
    n_AB, mu_AB, M_AB = maicos.lib.math.combine_subsample_variance(
        n_B, n_A, mu_B, mu_A, M_B, M_A
    )

    assert n_AB == len(series_A)
    assert mu_AB == mu_A
    assert M_AB == M_A

    # Test both series are empty
    n_AB, mu_AB, M_AB = maicos.lib.math.combine_subsample_variance(
        0, 0, np.nan, np.nan, np.nan, np.nan
    )

    assert n_AB == 0
    assert np.isnan(mu_AB)
    assert np.isnan(M_AB)
