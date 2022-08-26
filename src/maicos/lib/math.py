#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper functions for mathematical and physical operations."""

import numpy as np
from scipy import constants

from . import tables
from ._cutil import compute_structure_factor  # noqa: F401


# Max variation from the mean dt or dk that is allowed (~1e-10 suggested)
dt_dk_tolerance = 1e-8


def FT(t, x, indvar=True):
    """Discrete fast fourier transform.

    Takes the time series and the function as arguments.
    By default, returns the FT and the frequency:
    setting indvar=False means the function returns only the FT.
    """
    a, b = np.min(t), np.max(t)
    dt = (t[-1] - t[0]) / float(len(t) - 1)  # timestep
    if (abs((t[1:] - t[:-1] - dt)) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(t)
    # calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dt) * 2 * np.pi)
    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf2 = xf * (b - a) / N * np.exp(-1j * k * a)
    if indvar:
        return k, xf2
    else:
        return xf2


def iFT(k, xf, indvar=True):
    """Inverse discrete fast fourier transform.

    Takes the frequency series and the function as arguments.
    By default, returns the iFT and the time series:\
    setting indvar=False means the function returns only the iFT.
    """
    dk = (k[-1] - k[0]) / float(len(k) - 1)  # timestep
    if (abs((k[1:] - k[:-1] - dk)) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(k)
    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk)) * 2 * np.pi
    if N % 2 == 0:
        x2 = x * np.exp(-1j * t * N * dk / 2.) * N * dk / (2 * np.pi)
    else:
        x2 = x * np.exp(-1j * t * (N - 1) * dk / 2.) * N * dk / (2 * np.pi)
    if indvar:
        return t, x2
    else:
        return x2


def correlation(a, b=None, subtract_mean=False):
    """Calculate correlation or autocorrelation.

    Uses fast fourier transforms to give the correlation function
    of two arrays, or, if only one array is given, the autocorrelation.
    Setting subtract_mean=True causes the mean to be subtracted from
    the input data.
    """
    meana = int(subtract_mean) * np.mean(
        a)  # essentially an if statement for subtracting mean
    a2 = np.append(a - meana,
                   np.zeros(2**int(np.ceil((np.log(len(a)) / np.log(2))))
                            - len(a)))  # round up to a power of 2
    data_a = np.append(a2,
                       np.zeros(len(a2)))  # pad with an equal number of zeros
    fra = np.fft.fft(data_a)  # FT the data
    if b is None:
        sf = np.conj(
            fra
            ) * fra  # take the conj and multiply pointwise if autocorrelation
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(
            b - meanb,
            np.zeros(2**int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    cor = np.real(np.fft.ifft(sf)[:len(a)]) / np.array(range(
        len(a), 0, -1))  # inverse FFT and normalization
    return cor


def scalar_prod_corr(a, b=None, subtract_mean=False):
    """Give the corr. function of the scalar product of two vector timeseries.

    Arguments should be given in the form a[t, i],
    where t is the time variable along which the correlation is calculated,
    and i indexes the vector components.
    """
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], b[:, i], subtract_mean)

    return corr


def correlation_time(x_n, method='Sokal', c=8, mintime=3):
    """Compute the integrated correlation time of a timeseries.

    Parameters
    ----------
    x_n : numpy.ndarray, float
        timeseries
    method : str
        Method to choose integration cutoff Should be one of
        'Sokal'
        'Chodera'
    c : float
        cut-off factor for calculation of correlation time tau for Sokal method.
        cut-off T for integration is determined to be T >= c * tau
    mintime: int
        minimum possible value for cut-off

    Returns
    -------
    tau : float
        integrated correlation time
    """
    corr = correlation(x_n, subtract_mean=True)

    if method == 'Sokal':

        cutoff = tau = mintime
        for cutoff in range(mintime, len(x_n)):
            tau = np.sum((1 - np.arange(1, cutoff) / len(x_n))
                         * corr[1:cutoff] / corr[0])
            if cutoff > tau * c:
                break

            if cutoff > len(x_n) / 3:
                return -1

    if method == 'Chodera':

        cutoff = max(mintime, np.min(np.argwhere(corr < 0)))
        tau = np.sum((1 - np.arange(1, cutoff) / len(x_n))
                     * corr[1:cutoff] / corr[0])

    return tau


def new_mean(old_mean, data, length):
    r"""Compute the arithmetic mean of a series iteratively.

    Compute the arithmetic mean of n samples based on an
    existing mean of n-1 and the n-th value.

    Given the mean of a data series

    .. math::

        \bar x_N = \frac{1}{N} \sum_{n=1}^N x_n

    we seperate the last value

    .. math::

        \bar x_N = \frac{1}{N} \sum_{n=1}^{N-1} x_n + \frac{x_N}{N}

    and multiply 1 = (N - 1)/(N - 1)

    .. math::

        \bar x_N = \frac{N-1}{N} \frac{1}{N-1} \\
        \sum_{n=1}^{N-1} x_n + \frac{x_N}{N}

    The first term can be identified as the mean of the first N - 1 values
    and we arrive at
    .. math::

        \bar x_N = \frac{N-1}{N} \bar x_{N-1} + \frac{x_N}{N}


    Parameters
    ----------
    old_mean : float
        arithmetic mean of the first n - 1 samples.
    data : float
        n-th value of the series.
    length : int
        Length of the updated series, here called n.

    Returns
    -------
    new_mean : float
        Updated mean of the series of n values.

    Examples
    --------
    The mean of a data set can easily be calculated from the data points.
    However this requires one to keep all data points on hand until the
    end of the calculation.
    >>> np.mean([1,3,5,7])
    4.0
    Alternatively, one can update an existing mean, this requires only
    knowledge of the total number of samples.
    >>> maicos.utils.new_mean(np.mean([1, 3, 5]), 7, 4)
    4.0
    """
    return ((length - 1) * old_mean + data) / length


def new_variance(old_variance, old_mean, new_mean, data, length):
    """Calculate the variance of a timeseries iteratively.

    The variance of a timeseries can be calculated iteratively by
    using the following formula

    .. math::

        S_n = S_n-1 + (n-1) * (data_n - mean_n-1)^2 / (n-1)

    Floating point imprecision can lead to slight negative variances
    leading non defined standard deviations. Therefore a negetaive variance
    is set to 0.

    Parameters
    ----------
    old_variance : float
        The variance of the first n-1 samples.
    old_mean : float
        The mean of the first n-1 samples.
    new_mean : folat
        The mean of the full n samples.
    data : float
        The n-th value of the series.
    length : int
        Length of the updated series, here called n.

    Returns
    -------
    new_variance : float
        Updated variance of the series of n values.

    Examples
    --------
    The data set [1,5,5,1] has a variance of 4.0
    >>> np.var([1,5,5,1])
    4.0
    Knowing the total number of data points, this operation
    can be performed iteratively.
    >>> maicos.utils.new_variance(np.var([1,5,5]), 1, 4)
    4.0
    """
    S_old = old_variance * (length - 1)
    S_new = S_old + (data - old_mean) * (data - new_mean)

    if type(S_new) == np.ndarray:
        S_new[S_new < 0] = 0
    else:
        if S_new < 0:
            S_new = 0

    return S_new / length


def cluster_com(ag):
    """Calculate the center of mass of the atomgroup.

    Parameters
    ----------
    ag : mda.AtomGroup
        Group of atoms to calculate the center of mass for.

    Returns
    -------
    com : numpy.ndarray
        The center of mass.

    Without proper treatment of periodic boundrary conditions most algorithms
    will result in wrong center of mass calculations where molecules or clusters
    of particles are broken over the boundrary.

    Example ::

       +-----------+
       |           |
       | 1   x   2 |
       |           |
       +-----------+

    Following

    Linge Bai & David Breen (2008)
    Calculating Center of Mass in an Unbounded 2D Environment,
    Journal of Graphics Tools, 13:4, 53-60,
    DOI: 10.1080/2151237X.2008.10129266

    the coordinates of the particles are projected on a circle and weighted by
    their mass in this two dimensional space. The center of mass is obtained by
    transforming this point back to the corresponding point in the real system.
    This is done seperately for each dimension.

    Reasons for doing this include the analysis of clusters in periodic
    boundrary conditions and consistent center of mass calculation across
    box boundraries. This procedure results in the right center of mass
    as seen below ::

       +-----------+
       |           |
       x 1       2 |
       |           |
       +-----------+
    """
    L = ag.universe.dimensions[:3]
    theta = (ag.positions / L) * 2 * np.pi
    xi = ((np.cos(theta) * ag.masses[:, np.newaxis]).sum(axis=0)
          / ag.masses.sum())
    zeta = ((np.sin(theta) * ag.masses[:, np.newaxis]).sum(axis=0)
            / ag.masses.sum())
    theta_com = np.arctan2(-zeta, -xi) + np.pi
    com = theta_com / (2 * np.pi) * L
    return com


def symmetrize(m, axis=None, inplace=False):
    """Symmeterize an array.

    The shape of the array is preserved, but the elements are symmetrized
    with respect to the given axis.

    Parameters
    ----------
    m : array_like
        Input array to symmetrize
    axis : None or int or tuple of ints
         Axis or axes along which to symmetrize over. The default,
         axis=None, will symmetrize over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.
         If axis is a tuple of ints, symmetrizing is performed on all of the
         axes specified in the tuple.
    inplace : bool
        Do symmetrizations inplace. If `False` a new array is returnd.

    Returns
    -------
    out : array_like
        the symmetrized array

    Notes
    -----
    symmetrize uses :meth:`np.flip` for flipping the indices.

    Examples
    --------
    >>> A = np.arange(10).astype(float)
    >>> A
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> maicos.utils.symmetrize(A)
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])
    >>> maicos.utils.symmetrize(A, inplace=True)
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])
    >>> A
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])

    It also works for arrays with more than 1 dimensions in a
    general dimension.

    >>> A = np.arange(20).astype(float).reshape(2,10).T
    >>> A
    array([[ 0., 10.],
        [ 1., 11.],
        [ 2., 12.],
        [ 3., 13.],
        [ 4., 14.],
        [ 5., 15.],
        [ 6., 16.],
        [ 7., 17.],
        [ 8., 18.],
        [ 9., 19.]])
    >>> maicos.utils.symmetrize(A)
    array([[9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5],
        [9.5, 9.5]])
    >>> maicos.utils.symmetrize(A, axis=0)
    array([[ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5],
        [ 4.5, 14.5]])
    """
    if inplace:
        out = m
    else:
        out = np.copy(m)

    out += np.flip(m, axis=axis)
    out /= 2

    return out


def compute_form_factor(q, atom_type):
    """Calculate the form factor for the given element for given q (1/Å).

    Handles united atom types like CH4 etc ...
    """
    element = tables.atomtypes[atom_type]

    if element == "CH1":
        form_factor = compute_form_factor(q, "C") + compute_form_factor(q, "H")
    elif element == "CH2":
        form_factor = compute_form_factor(
            q, "C") + 2 * compute_form_factor(q, "H")
    elif element == "CH3":
        form_factor = compute_form_factor(
            q, "C") + 3 * compute_form_factor(q, "H")
    elif element == "CH4":
        form_factor = compute_form_factor(
            q, "C") + 4 * compute_form_factor(q, "H")
    elif element == "NH1":
        form_factor = compute_form_factor(q, "N") + compute_form_factor(q, "H")
    elif element == "NH2":
        form_factor = compute_form_factor(
            q, "N") + 2 * compute_form_factor(q, "H")
    elif element == "NH3":
        form_factor = compute_form_factor(
            q, "N") + 3 * compute_form_factor(q, "H")
    else:
        form_factor = tables.CM_parameters[element].c
        q2 = (q / (4 * np.pi))**2
        for i in range(4):
            form_factor += tables.CM_parameters[element].a[i] * \
                np.exp(-tables.CM_parameters[element].b[i] * q2)

    return form_factor


def mu(rho, temperature, m):
    """Calculate the chemical potential.

    The chemical potential is calculated from the
    density: mu = k_B T log(rho. / m)
    """
    # kT in KJ/mol
    kT = temperature * constants.Boltzmann \
        * constants.Avogadro / constants.kilo

    results = []

    for srho, mass in zip(np.array(rho).T, m):
        # De Broglie (converted to nm)
        db = np.sqrt(
            constants.h ** 2 / (2 * np.pi * mass * constants.atomic_mass
                                * constants.Boltzmann * temperature)
            ) / constants.angstrom

        if np.all(srho > 0):
            results.append(kT * np.log(srho * db ** 3))
        elif np.any(srho == 0):
            results.append(np.float64("-inf") * np.ones(srho.shape))
        else:
            results.append(np.float64("nan") * np.ones(srho.shape))
    return np.squeeze(np.array(results).T)


def dmu(rho, drho, temperature):
    """Calculate the error of the chemical potential.

    The error is calculated from the density using propagation of uncertainty.
    """
    kT = temperature * constants.Boltzmann \
        * constants.Avogadro / constants.kilo

    results = []

    for srho, sdrho in zip(np.array(rho).T, np.array(drho).T):
        if np.all(srho > 0):
            results.append(kT * (sdrho / srho))
        else:
            results.append(np.float64("nan") * np.ones(srho.shape))
    return np.squeeze(np.array(results).T)
