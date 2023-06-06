#!/usr/bin/env python -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8
# -*-
#
# Copyright (c) 2023 Authors and contributors (see the AUTHORS.rst file for the full
# list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper functions for mathematical and physical operations."""
from typing import Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np

from . import tables
from ._cutil import compute_structure_factor  # noqa: F401


# Max variation from the mean dt or dk that is allowed (~1e-10 suggested)
dt_dk_tolerance = 1e-8


def FT(
    t: np.ndarray, x: np.ndarray, indvar: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Discrete Fast Fourier Transform (FFT).

    Parameters
    ----------
    t : numpy.ndarray
        Time values of the time series.
    x : numpy.ndarray
        Function values corresponding to the time series.
    indvar : bool
        If :obj:`True`, returns the FFT and frequency values. If :obj:`False`, returns
        only the FFT. Default is :obj:`True`.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        If indvar is :obj:`True`, returns a tuple (k, xf2) where:
            - k (numpy.ndarray): Frequency values corresponding to the FFT.
            - xf2 (numpy.ndarray): FFT of the input function, scaled by the time range
              and phase shifted.

        If indvar is :obj:`False`, returns the FFT (xf2) directly as a
        :class:`numpy.ndarray`.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    Example
    -------
    >>> t = np.linspace(0, np.pi, 4)
    >>> x = np.sin(t)
    >>> k, xf2 = FT(t, x)
    >>> k
    array([-3. , -1.5,  0. ,  1.5])
    >>> np.round(xf2, 2)
    array([ 0.  +0.j  , -0.68+0.68j,  1.36+0.j  , -0.68-0.68j])

    See Also
    --------
    :func:`iFT` : For the inverse fourier transform.
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


def iFT(
    k: np.ndarray, xf: np.ndarray, indvar: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Inverse discrete fast fourier transform.

    Takes the frequency series and the function as arguments. By default, returns the
    iFT and the time series. Setting indvar=False means the function returns only the
    iFT.

    Parameters
    ----------
    k : numpy.ndarray
        The frequency series.
    xf : numpy.ndarray
        The function series in the frequency domain.
    indvar : bool
        If :obj:`True`, return both the iFT and the time series. If :obj:`False`, return
        only the iFT. Default is :obj:`True`.

    Returns
    -------
    tuple(numpy.ndarray, np.ndarray) or numpy.ndarray
        If indvar is :obj:`True`, returns a tuple containing the time series and the
        iFT. If indvar is :obj:`False`, returns only the iFT.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    See Also
    --------
    :func:`FT` : For the fourier transform.
    """
    dk = (k[-1] - k[0]) / float(len(k) - 1)  # timestep
    if (abs((k[1:] - k[:-1] - dk)) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(k)
    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk)) * 2 * np.pi
    if N % 2 == 0:
        x2 = x * np.exp(-1j * t * N * dk / 2.0) * N * dk / (2 * np.pi)
    else:
        x2 = x * np.exp(-1j * t * (N - 1) * dk / 2.0) * N * dk / (2 * np.pi)
    if indvar:
        return t, x2
    else:
        return x2


def correlation(
    a: np.ndarray, b: Optional[np.ndarray] = None, subtract_mean: bool = False
) -> np.ndarray:
    """Calculate correlation or autocorrelation.

    Uses fast fourier transforms to give the correlation function of two arrays, or, if
    only one array is given, the autocorrelation. Setting ``subtract_mean=True`` causes
    the mean to be subtracted from the input data.

    Parameters
    ----------
    a : numpy.ndarray
        The first input array to calculate the correlation
    b : numpy.ndarray
        The second input array. If :obj:`None`, autocorrelation of ``a`` is calculated.
        Default is :obj:`None`.
    subtract_mean : bool, optional
        If obj:`True`, subtract the mean from the input data.

    Returns
    -------
    np.ndarray
        The correlation or autocorrelation function.
    """
    meana = int(subtract_mean) * np.mean(
        a
    )  # essentially an if statement for subtracting mean
    a2 = np.append(
        a - meana, np.zeros(2 ** int(np.ceil((np.log(len(a)) / np.log(2)))) - len(a))
    )  # round up to a power of 2
    data_a = np.append(a2, np.zeros(len(a2)))  # pad with an equal number of zeros
    fra = np.fft.fft(data_a)  # FT the data
    if b is None:
        sf = (
            np.conj(fra) * fra
        )  # take the conj and multiply pointwise if autocorrelation
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(
            b - meanb,
            np.zeros(2 ** int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)),
        )
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    cor = np.real(np.fft.ifft(sf)[: len(a)]) / np.array(
        range(len(a), 0, -1)
    )  # inverse FFT and normalization
    return cor


def scalar_prod_corr(
    a: np.ndarray, b: Optional[np.ndarray] = None, subtract_mean: bool = False
) -> np.ndarray:
    """Give the corr. function of the scalar product of two vector timeseries.

    Arguments should be given in the form a[t, i], where t is the time variable along
    which the correlation is calculated, and i indexes the vector components.

    Parameters
    ----------
    a : numpy.ndarray
        The first vector timeseries of shape (t, i).
    b : numpy.ndarray
        The second vector timeseries of shape (t, i). If :obj:`None`, correlation with
        itself is calculated. Default is :obj:`None`.
    subtract_mean : bool
        If :obj:`True`, subtract the mean from the timeseries before calculating the
        correlation. Default is :obj:`False`.

    Returns
    -------
    np.ndarray
        The correlation function of the scalar product of the vector timeseries.

    """
    corr = np.zeros(len(a[:, 0]))

    if b is None:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], None, subtract_mean)

    else:
        for i in range(0, len(a[0, :])):
            corr[:] += correlation(a[:, i], b[:, i], subtract_mean)

    return corr


def correlation_time(
    x_n: np.ndarray, method: str = "sokal", c: float = 8, mintime: int = 3
) -> float:
    r"""Compute the integrated correlation time of a timeseries.

    The "sokal" method is based on :footcite:t:`sokalLecture`. The "chodera" method is
    based on :footcite:t:`choderaWeightedHistogramAnalysis2007`.

    Parameters
    ----------
    x_n : numpy.ndarray
        timeseries
    method : str, {"sokal", "chodera"}
        Method to choose integration cutoff. Default is "sokal".
    c : float
        cut-off factor for calculation of correlation time :math:`\tau` for the Sokal
        method. The cut-off :math:`T` for integration is determined to be :math:`T >= c
        \cdot tau`. Default is 8.
    mintime: int
        minimum possible value for cut-off. Default is 3.

    Returns
    -------
    tau : float
        integrated correlation time

    Raises
    ------
    ValueError
        If method is not one of "sokal" or "chodera"

    References
    ----------
    .. footbibliography::
    """
    corr = correlation(x_n, subtract_mean=True)

    if method == "sokal":
        cutoff = mintime
        tau = mintime
        for cutoff in range(mintime, len(x_n)):
            tau = np.sum(
                (1 - np.arange(1, cutoff) / len(x_n)) * corr[1:cutoff] / corr[0]
            )
            if cutoff > tau * c:
                break

            if cutoff > len(x_n) / 3:
                return -1

    elif method == "chodera":
        cutoff = np.max([mintime, np.min(np.argwhere(corr < 0))])
        tau = np.sum((1 - np.arange(1, cutoff) / len(x_n)) * corr[1:cutoff] / corr[0])
    else:
        raise ValueError(
            f"Unknown method: {method}. " "Chose either 'sokal' or 'chodera'."
        )
    return tau


def new_mean(old_mean: float, data: float, length: int) -> float:
    r"""Compute the arithmetic mean of a series iteratively.

    Compute the arithmetic mean of n samples based on an existing mean of n-1 and the
    n-th value.

    Given the mean of a data series

    .. math::

        \bar x_N = \frac{1}{N} \sum_{n=1}^N x_n

    we seperate the last value

    .. math::

        \bar x_N = \frac{1}{N} \sum_{n=1}^{N-1} x_n + \frac{x_N}{N}

    and multiply 1 = (N - 1)/(N - 1)

    .. math::

        \bar x_N = \frac{N-1}{N} \frac{1}{N-1} \\ \sum_{n=1}^{N-1} x_n + \frac{x_N}{N}

    The first term can be identified as the mean of the first N - 1 values and we arrive
    at

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
    The mean of a data set can easily be calculated from the data points. However this
    requires one to keep all data points on hand until the end of the calculation.

    >>> np.mean([1, 3, 5, 7])
    4.0

    Alternatively, one can update an existing mean, this requires only knowledge of the
    total number of samples.

    >>> new_mean(np.mean([1, 3, 5]), data=7, length=4)
    4.0
    """
    return ((length - 1) * old_mean + data) / length


def new_variance(
    old_variance: Union[float, np.ndarray],
    old_mean: Union[float, np.ndarray],
    new_mean: Union[float, np.ndarray],
    data: Union[float, np.ndarray],
    length: int,
) -> Union[float, np.ndarray]:
    r"""Calculate the variance of a timeseries iteratively.

    The variance of a timeseries :math:`x_n` can be calculated iteratively by using the
    following formula:

    .. math::

        S_n = S_n-1 + (n-1) * (x_n - \bar{x}_n-1)^2 / (n-1)

    Here, :math:`\bar{x}_n` is the mean of the timeseries up to the :math:`n`-th value.

    Floating point imprecision can lead to slight negative variances leading non defined
    standard deviations. Therefore a negetaive variance is set to 0.

    Parameters
    ----------
    old_variance : float, numpy.ndarray
        The variance of the first n-1 samples.
    old_mean : float
        The mean of the first n-1 samples.
    new_mean : float, numpy.ndarray
        The mean of the full n samples.
    data : float, numpy.ndarray
        The n-th value of the series.
    length : int
        Length of the updated series, here called n.

    Returns
    -------
    new_variance : float
        Updated variance of the series of n values.

    Examples
    --------
    The data set ``[1, 5, 5, 1]`` has a variance of ``4.0``

    >>> np.var([1, 5, 5, 1])
    4.0

    Knowing the total number of data points, this operation can be performed
    iteratively.

    >>> new_variance(
    ...     old_variance=np.var([1, 5, 5]),
    ...     old_mean=np.mean([1, 5, 5]),
    ...     new_mean=np.mean([1, 5, 5, 1]),
    ...     data=1,
    ...     length=4,
    ... )
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


def center_cluster(ag: mda.AtomGroup, weights: np.ndarray) -> np.ndarray:
    """Calculate the center of the atomgroup with respect to some weights.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        Group of atoms to calculate the center for.

    weights : numpy.ndarray
        Weights in the shape of ag.

    Returns
    -------
    com : numpy.ndarray
        The center with respect to the weights.


    Without proper treatment of periodic boundrary conditions most algorithms will
    result in wrong center of mass calculations where molecules or clusters of particles
    are broken over the boundrary.

    Example ::

       +-----------+
       |           |
       | 1   x   2 |
       |           |
       +-----------+

    Following

    Linge Bai & David Breen (2008) Calculating Center of Mass in an Unbounded 2D
    Environment, Journal of Graphics Tools, 13:4, 53-60, DOI:
    10.1080/2151237X.2008.10129266

    the coordinates of the particles are projected on a circle and weighted by their
    mass in this two dimensional space. The center of mass is obtained by transforming
    this point back to the corresponding point in the real system. This is done
    seperately for each dimension.

    Reasons for doing this include the analysis of clusters in periodic boundrary
    conditions and consistent center of mass calculation across box boundraries. This
    procedure results in the right center of mass as seen below ::

       +-----------+
       |           |
       x 1       2 | |           | +-----------+
    """
    theta = (ag.positions / ag.universe.dimensions[:3]) * 2 * np.pi
    xi = (np.cos(theta) * weights[:, None]).sum(axis=0) / weights.sum()
    zeta = (np.sin(theta) * weights[:, None]).sum(axis=0) / weights.sum()
    theta_com = np.arctan2(-zeta, -xi) + np.pi
    return theta_com / (2 * np.pi) * ag.universe.dimensions[:3]


def symmetrize(
    m: np.ndarray, axis: Union[None, int, Tuple[int]] = None, inplace: bool = False
) -> np.ndarray:
    """Symmeterize an array.

    The shape of the array is preserved, but the elements are symmetrized with respect
    to the given axis.

    Parameters
    ----------
    m : array_like
        Input array to symmetrize
    axis : int, tuple(int)
         Axis or axes along which to symmetrize over. The default, ``axis=None``, will
         symmetrize over all of the axes of the input array. If axis is negative it
         counts from the last to the first axis. If axis is a :obj:`tuple` of ints,
         symmetrizing is performed on all of the axes specified in the :obj:`tuple`.
    inplace : bool
        Do symmetrizations inplace. If :obj:`False` a new array is returned.

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
    >>> symmetrize(A)
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])
    >>> symmetrize(A, inplace=True)
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])
    >>> A
    array([4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5])

    It also works for arrays with more than 1 dimensions in a general dimension.

    >>> A = np.arange(20).astype(float).reshape(2, 10).T
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
    >>> symmetrize(A)
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
    >>> symmetrize(A, axis=0)
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


def compute_form_factor(q: float, atom_type: str) -> float:
    r"""Calculate the form factor :math:`f(q)`.

    :math:`f(q)` is expressed in terms of the scattering vector as

    .. math::
        f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c \,.

    The coefficients :math:`a_{1,\dots,4}`, :math:`b_{1,\dots,4}` and :math:`c` are also
    known as Cromer-Mann X-ray scattering factors and are documented in
    :footcite:t:`princeInternationalTablesCrystallography2004`.

    For determining the elements :obj:`maicos.lib.tables.atomtypes` is used and the
    Cromer-Mann X-ray scattering factors are stored in
    :obj:`maicos.lib.tables.CM_parameters`.

    Parameters
    ----------
    q : float
        The magnitude of the scattering vector in reciprocal angstroms (1/Å).
    atom_type : str
        The type of the atom for which the form factor is calculated.

    Returns
    -------
    float
        The calculated form factor for the specified atom type and q.
    """
    element = tables.atomtypes[atom_type]

    if element == "CH1":
        form_factor = compute_form_factor(q, "C") + compute_form_factor(q, "H")
    elif element == "CH2":
        form_factor = compute_form_factor(q, "C") + 2 * compute_form_factor(q, "H")
    elif element == "CH3":
        form_factor = compute_form_factor(q, "C") + 3 * compute_form_factor(q, "H")
    elif element == "CH4":
        form_factor = compute_form_factor(q, "C") + 4 * compute_form_factor(q, "H")
    elif element == "NH1":
        form_factor = compute_form_factor(q, "N") + compute_form_factor(q, "H")
    elif element == "NH2":
        form_factor = compute_form_factor(q, "N") + 2 * compute_form_factor(q, "H")
    elif element == "NH3":
        form_factor = compute_form_factor(q, "N") + 3 * compute_form_factor(q, "H")
    else:
        form_factor = tables.CM_parameters[element].c
        q2 = (q / (4 * np.pi)) ** 2
        for i in range(4):
            form_factor += tables.CM_parameters[element].a[i] * np.exp(
                -tables.CM_parameters[element].b[i] * q2
            )

    return form_factor
