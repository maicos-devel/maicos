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
from data import SPCE_GRO, SPCE_ITP, WATER_GRO_NPT, WATER_TPR_NPT  # noqa: E402


class Test_sfactor:
    """Tests for the sfactor."""

    @pytest.fixture
    def ag(self):
        """Import MDA universe."""
        u = mda.Universe(WATER_TPR_NPT, WATER_GRO_NPT)
        return u.atoms

    @pytest.fixture
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

    @pytest.mark.parametrize("qmin", [0, 0.05])
    @pytest.mark.parametrize("qmax", [0.075, 0.1])
    def test_sfactor(self, ag, qS, qmin, qmax):
        """Test sfactor."""
        q, S = maicos.lib.math.structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            qmin,
            qmax,
            0,  # mimtheta
            np.pi,  # thetamax
            np.ones(len(ag.positions)),
        )

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        # Get indices to slice qS array
        sel_indices = np.logical_and(qmin < qS[0], qS[0] < qmax)

        assert_allclose(q, qS[0][sel_indices], rtol=1e-2)

        # Only check S for full q width
        if qmin == 0 and qmax == 1:
            assert_allclose(S, qS[1], rtol=1e-2)

    def test_sfactor_angle(self, ag):
        """Test sfactor angle."""
        q, S = maicos.lib.math.structure_factor(
            np.double(ag.positions),
            np.double(ag.universe.dimensions)[:3],
            0,  # qmin
            0.5,  # qmax
            np.pi / 4,  # thetamin
            np.pi / 2,  # thetamax
            np.ones(len(ag.positions)),
        )

        q = q.flatten()
        S = S.flatten()
        nonzeros = np.where(S != 0)[0]

        q = q[nonzeros]
        S = S[nonzeros]

        sorted_ind = np.argsort(q)
        q = q[sorted_ind]
        S = S[sorted_ind]

        assert_allclose(q, np.array([0.25, 0.25, 0.36, 0.36, 0.36, 0.44]), rtol=1e-1)
        assert_allclose(
            S, np.array([6.91, 835.3, 192.54, 157.79, 506.76, 99.65]), rtol=1e-1
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
    assert_allclose(abs(t[np.argmax(sin_FT)]), 5, rtol=1e-2)


def test_FT_unequal_spacing():
    """Tests for the Fourier transform with unequal spacing."""
    t = np.linspace(-np.pi, np.pi, 500)
    t[0] += 1e-5  # make it unequal
    sin = np.sin(5 * t)
    match = "Time series not equally spaced!"
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.FT(t, sin)


def test_iFT():
    """Tests for the inverse Fourier transform."""
    x = np.linspace(-np.pi, np.pi, 500)
    sin = np.sin(5 * x)
    t, sin_FT = maicos.lib.math.FT(x, sin)
    sin_new = maicos.lib.math.iFT(t, sin_FT, indvar=False)
    # Shift to positive y domain to avoid comparing 0
    assert_allclose(2 + sin, 2 + sin_new.real, rtol=1e-1)


def test_iFT_unequal_spacing():
    """Tests for the inverse Fourier transform with unequal spacing."""
    t = np.linspace(-np.pi, np.pi, 500)
    t[0] += 1e-5  # make it unequal
    sin = np.sin(5 * t)
    match = "Time series not equally spaced!"
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.iFT(t, sin)


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


def test_form_factor():
    """Regression test for the atomic form factor as function q.

    Reference values for hydrogen are taken from Table 6.1.1.1 in
    https://it.iucr.org/Cb/ch6o1v0001/
    """
    reference_values = np.array(
        [
            [0.00, 1.000],
            [0.01, 0.998],
            [0.02, 0.991],
            [0.03, 0.980],
            [0.04, 0.966],
            [0.05, 0.947],
            [0.06, 0.925],
            [0.07, 0.900],
            [0.08, 0.872],
            [0.09, 0.842],
            [0.10, 0.811],
            [0.22, 0.424],
            [0.46, 0.090],
        ]
    )

    sin_theta = reference_values[:, 0]
    q = 4 * np.pi * sin_theta
    desired = reference_values[:, 1]

    assert_allclose(
        actual=maicos.lib.math.atomic_form_factor(q, "H"),
        desired=desired,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    ("atom_type", "n_electrons"),
    [
        ("C", 6),
        ("Cval", 6),
        ("CVAL", 6),  # upper case elements should also work
        ("O", 8),
        ("CH1", 7),
        ("CH2", 8),
        ("CH3", 9),
        ("CH4", 10),
        ("NH1", 8),
        ("NH2", 9),
        ("NH3", 10),
    ],
)
def test_form_factor_zero(atom_type, n_electrons):
    """Test that the atomic form factor for q=0 is same as the number of electrons."""
    assert_allclose(
        actual=maicos.lib.math.atomic_form_factor(0.0, atom_type),
        desired=n_electrons,
        rtol=1e-3,
    )


def test_form_factor_unknown_element():
    """Test that an unknown elements raise an error."""
    match = (
        "Element 'foo' not found. Known elements are listed in the "
        "`maicos.lib.tables.elements` set."
    )
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.atomic_form_factor(0.0, "foo")


def test_rdf_structure_factor_unequal_spacing():
    """Test that a ValueError is raised if the input is not equally spaced.

    Additional tests for the functionality are located in `test_saxs.py` and
    `test_diorderstructurefactor.py`.
    """
    r = np.linspace(0, 10, 10)
    r[0] += 1e-5

    rdf = np.ones(len(r))
    density = 1.0
    match = "Distance array `r` is not equally spaced!"
    with pytest.raises(ValueError, match=match):
        maicos.lib.math.rdf_structure_factor(rdf=rdf, r=r, density=density)
