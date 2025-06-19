# -*-
#
# Copyright (c) 2025 Authors and contributors (see the AUTHORS.rst file for the full
# list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helper functions for mathematical and physical operations."""

import MDAnalysis as mda
import numpy as np
from scipy.fftpack import dst

from . import tables
from ._cmath import compute_structure_factor  # noqa: F401

# Max variation from the mean dt or dk that is allowed (~1e-10 suggested)
dt_dk_tolerance = 1e-8


def FT(
    t: np.ndarray, x: np.ndarray, indvar: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Discrete Fourier transformation using fast Fourier transformation (FFT).

    Parameters
    ----------
    t : numpy.ndarray
        Time values of the time series.
    x : numpy.ndarray
        Function values corresponding to the time series.
    indvar : bool
        If :obj:`True`, returns the FFT and frequency values. If :obj:`False`, returns
        only the FFT.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        If ``indvar`` is :obj:`True`, returns a tuple ``(k, xf2)`` where:
            - ``k`` (numpy.ndarray): Frequency values corresponding to the FFT.
            - ``xf2`` (numpy.ndarray): FFT of the input function, scaled by the time
              range and phase shifted.

        If indvar is :obj:`False`, returns the FFT (``xf2``) directly as a
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
    if (abs(t[1:] - t[:-1] - dt) > dt_dk_tolerance).any():
        raise RuntimeError("Time series not equally spaced!")
    N = len(t)

    # calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N, d=dt) * 2 * np.pi)

    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf2 = xf * (b - a) / N * np.exp(-1j * k * a)

    if indvar:
        return k, xf2
    return xf2


def iFT(
    k: np.ndarray, xf: np.ndarray, indvar: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Inverse Fourier transformation using fast Fourier transformation (FFT).

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
        only the iFT.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray) or numpy.ndarray
        If indvar is :obj:`True`, returns a tuple containing the time series and the
        iFT. If indvar is :obj:`False`, returns only the iFT.

    Raises
    ------
    RuntimeError
        If the time series is not equally spaced.

    See Also
    --------
    :func:`FT` : For the Fourier transform.

    """
    dk = (k[-1] - k[0]) / float(len(k) - 1)  # timestep
    if (abs(k[1:] - k[:-1] - dk) > dt_dk_tolerance).any():
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
    return x2


def correlation(
    a: np.ndarray, b: np.ndarray | None = None, subtract_mean: bool = False
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
    subtract_mean : bool
        If :obj:`True`, subtract the mean from the input data.

    Returns
    -------
    numpy.ndarray
        The correlation or autocorrelation function.

    """
    meana = int(subtract_mean) * np.mean(
        a
    )  # essentially an if statement for subtracting mean
    a2 = np.append(
        a - meana, np.zeros(2 ** int(np.ceil(np.log(len(a)) / np.log(2))) - len(a))
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
            np.zeros(2 ** int(np.ceil(np.log(len(b)) / np.log(2))) - len(b)),
        )
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    return np.real(np.fft.ifft(sf)[: len(a)]) / np.array(
        range(len(a), 0, -1)
    )  # inverse FFT and normalization


def scalar_prod_corr(
    a: np.ndarray, b: np.ndarray | None = None, subtract_mean: bool = False
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
        itself is calculated.
    subtract_mean : bool
        If :obj:`True`, subtract the mean from the timeseries before calculating the
        correlation.

    Returns
    -------
    numpy.ndarray
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
    timeseries: np.ndarray,
    method: str = "sokal",
    mintime: int = 3,
    sokal_factor: float = 8,
) -> float:
    r"""Compute the integrated correlation time of a time series.

    The integrated correlation time (in units of the sampling interval) is given by

    .. math::
        \tau = \sum\limits_{t=1}^{N_\mathrm{cut}} C(t) \left(1 - \frac{t}{N}\right)

    where :math:`N_\mathrm{cut} < N` is a subset of the time series of length :math:`N`
    and :math:`C(t)` is the discrete-time autocorrelation function. To obtain the upper
    limit of the sum :math:`N_\mathrm{cut}` two different methods are provided:

    1. For "chodera" :footcite:p:`choderaWeightedHistogramAnalysis2007`
       :math:`N_\mathrm{cut}` is given by the time when :math:`C(t)`
       crosses zero the first time.

    2. For "sokal" :footcite:p:`sokalLecture` :math:`N_\mathrm{cut}` is determined
       iteratively by stepwise increasing until

       .. math::
            N_\mathrm{cut} \geq c \cdot \tau

       where :math:`c` is the constant ``sokal_factor``. If the condition is never
       fulfilled, ``-1`` is returned, indicating that the time series does not provide
       sufficient statistics to estimate a
       correlation time.

    While both methods give the same correlation time for a smooth time series that
    decays to 0, "sokal" will results in a more reasonable result for actual time series
    that are noisy and cross zero several times.

    Parameters
    ----------
    timeseries : numpy.ndarray
        The time series used to calculate the correlation time from.
    method : {``"sokal"``, ``"chodera"``}
        Method to choose summation cutoff :math:`N_\mathrm{cut}`.
    mintime: int
        Minimum possible value for :math:`N_\mathrm{cut}`.
    sokal_factor : float
        Cut-off factor :math:`c` for the Sokal method.

    Returns
    -------
    tau : float
        Integrated correlation time :math:`\tau`. If ``-1`` (only for
        ``method="sokal"``) the provided time series does not provide sufficient
        statistics to estimate a correlation time.

    Raises
    ------
    ValueError
        If mintime is larger than the length of the timeseries.
    ValueError
        If method is not one of "sokal" or "chodera".

    References
    ----------
    .. footbibliography::

    """
    if mintime > len(timeseries):
        raise ValueError(
            f"mintime ({mintime}) has to be smaller then the length of `timeseries` "
            f"({len(timeseries)})."
        )

    corr = correlation(timeseries, subtract_mean=True)

    if method == "sokal":
        for cutoff in range(mintime, len(timeseries)):
            tau = np.sum(
                (1 - np.arange(1, cutoff) / len(timeseries)) * corr[1:cutoff] / corr[0]
            )
            if cutoff >= sokal_factor * tau:
                break

            if cutoff > len(timeseries) / 3:
                return -1

    elif method == "chodera":
        cutoff = np.max([mintime, np.min(np.argwhere(corr < 0))])
        tau = np.sum(
            (1 - np.arange(1, cutoff) / len(timeseries)) * corr[1:cutoff] / corr[0]
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Chose either 'sokal' or 'chodera'."
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

    >>> print(np.mean([1, 3, 5, 7]))
    4.0

    Alternatively, one can update an existing mean, this requires only knowledge of the
    total number of samples.

    >>> print(new_mean(np.mean([1, 3, 5]), data=7, length=4))
    4.0

    """
    return ((length - 1) * old_mean + data) / length


def new_variance(
    old_variance: float | np.ndarray,
    old_mean: float | np.ndarray,
    new_mean: float | np.ndarray,
    data: float | np.ndarray,
    length: int,
) -> float | np.ndarray:
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

    >>> print(np.var([1, 5, 5, 1]))
    4.0

    Knowing the total number of data points, this operation can be performed
    iteratively.

    >>> print(
    ...     new_variance(
    ...         old_variance=np.var([1, 5, 5]),
    ...         old_mean=np.mean([1, 5, 5]),
    ...         new_mean=np.mean([1, 5, 5, 1]),
    ...         data=1,
    ...         length=4,
    ...     )
    ... )
    4.0

    """
    S_old = old_variance * (length - 1)
    S_new = S_old + (data - old_mean) * (data - new_mean)

    if type(S_new) is np.ndarray:
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


    Without proper treatment of periodic boundrary conditions (PBC) most algorithms will
    result in wrong center calculations. As shown below without treating PBC the center
    of mass is located in the center of the box ::

       +-----------+
       |           |
       | 1   x   2 |
       |           |
       +-----------+

    However, the distance accross the box boundary is shorter and therefore the center
    with PBC should be located somwhere else. The correct way to calculate the center is
    described in :footcite:t:`bai_calculating_2008` where coordinates of the particles
    are projected on a circle and weighted by their mass in this two dimensional space.
    The center of mass is obtained by transforming this point back to the corresponding
    point in the real system. This is done seperately for each dimension.

    Reasons for doing this include the analysis of clusters in periodic boundrary
    conditions and consistent center of mass calculation across box boundraries. This
    procedure results in the right center of mass as seen below ::

       +-----------+
       |           |
       x 1       2 |
       |           |
       +-----------+

    """
    theta = (ag.positions / ag.universe.dimensions[:3]) * 2 * np.pi
    xi = (np.cos(theta) * weights[:, None]).sum(axis=0) / weights.sum()
    zeta = (np.sin(theta) * weights[:, None]).sum(axis=0) / weights.sum()
    theta_com = np.arctan2(-zeta, -xi) + np.pi
    return theta_com / (2 * np.pi) * ag.universe.dimensions[:3]


def symmetrize(
    m: np.ndarray,
    axis: None | int | tuple[int] = None,
    inplace: bool = False,
    parity: str = "even",
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
    parity : str
        The parity to use for symmetrization. Can be either "even" or "odd".
        "even" parity means that the output array will be symmetric with respect to the
        specified axis, while "odd" parity means that the output array will be
        antisymmetric.

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

    Antisymmetrization can be achieved by setting ``parity="odd"``.
    >>> A = np.arange(10).astype(float)
    >>> A
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
    >>> symmetrize(A, parity="odd")
    array([-4.5, -3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5])

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
    # The returned array will be of type float
    out = m.copy().astype("float")
    out += (1 if parity == "even" else -1) * np.flip(m, axis=axis)
    out /= 2

    if inplace:
        # To safely cast the the original array type to float in-place,
        # first change the dtype to float...
        m.dtype = np.dtype("float")
        # ...and then write the new values to the original array.
        m[...] = out
        return m
    return out


def compute_form_factor(q: float, element: str) -> float:
    r"""Calculate the form factor :math:`f(q)`.

    :math:`f(q)` is expressed in terms of the scattering vector as

    .. math::
        f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c \,.

    The coefficients :math:`a_{1,\dots,4}`, :math:`b_{1,\dots,4}` and :math:`c` are also
    known as Cromer-Mann X-ray scattering factors and are documented in
    :footcite:t:`princeInternationalTablesCrystallography2004` and taken from the `TU
    Graz
    <https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_.
    and stored in :obj:`maicos.lib.tables.CM_parameters`.

    Parameters
    ----------
    q : float
        The magnitude of the scattering vector in reciprocal angstroms (1/Å).
    element : str
        The element for which the form factor is calculated. Known elements are listed
        in the :attr:`maicos.lib.tables.elements` set. United-atom models such as
        ``"CH1"``, ``"CH2"``, ``"CH3"``, ``"CH4"``, ``"NH1"``, ``"NH2"``, and ``"NH3"``
        are also supported.

        .. note::

            ``element`` is converted to title case to avoid most common issues with
            MDAnalysis which uses upper case elements by default. For example ``"MG"``
            will be converted to ``"Mg"``.

    Returns
    -------
    float
        The calculated form factor for the specified element and q in units of
        electrons.

    """
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
        if element.title() not in tables.CM_parameters:
            raise ValueError(
                f"Element '{element}' not found in Cromer-Mann parameters. Known "
                "elements are listed in the `maicos.lib.tables.elements` set."
            )
        element = element.title()
        form_factor = tables.CM_parameters[element].c
        # q / (4 * pi) = sin(theta) / lambda
        q2 = (q / (4 * np.pi)) ** 2
        for i in range(4):
            form_factor += tables.CM_parameters[element].a[i] * np.exp(
                -tables.CM_parameters[element].b[i] * q2
            )

    return form_factor


def transform_cylinder(
    positions: np.ndarray, origin: np.ndarray, dim: int
) -> np.ndarray:
    """Transform positions into cylinder coordinates.

    The origin of th coordinate system is at `origin`, the direction of the cylinder is
    defined by `dim`.

    Parameters
    ----------
    positions : numpy.ndarray
        Cartesian coordinates (x,y,z)
    origin : numpy.ndarray
        Origin of the new cylindrical coordinate system (x,y,z).
    dim : int
        Direction of the cylinder axis (0=x, 1=y, 2=z).

    Returns
    -------
    numpy.ndarray
        Positions in cylinder coordinates (r, phi, z)

    """
    trans_positions = np.zeros(positions.shape)

    odims = np.roll(np.arange(3), -dim)[1:]

    # shift origin to box center
    pos_xyz_center = positions - origin

    # r component
    trans_positions[:, 0] = np.linalg.norm(pos_xyz_center[:, odims], axis=1)

    # phi component
    np.arctan2(*pos_xyz_center[:, odims].T, out=trans_positions[:, 1])

    # z component
    trans_positions[:, 2] = np.copy(positions[:, dim])

    return trans_positions


def transform_sphere(positions: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Transform positions into spherical coordinates.

    The origin of the new coordinate system is at `origin`.

    Parameters
    ----------
    positions : numpy.ndarray
        Cartesian coordinates (x,y,z)
    origin : numpy.ndarray
        Origin of the new spherical coordinate system (x,y,z).

    Returns
    -------
    numpy.ndarray
        Positions in spherical coordinates (:math:`r`, phi, theta)

    """
    trans_positions = np.zeros(positions.shape)

    # shift origin to box center
    # positions -= origin
    pos_xyz_center = positions - origin

    # r component
    trans_positions[:, 0] = np.linalg.norm(pos_xyz_center, axis=1)
    # phi component
    np.arctan2(pos_xyz_center[:, 1], pos_xyz_center[:, 0], out=trans_positions[:, 1])
    # theta component
    np.arccos(pos_xyz_center[:, 2] / trans_positions[:, 0], out=trans_positions[:, 2])

    return trans_positions


def compute_rdf_structure_factor(
    rdf: np.ndarray, r: np.ndarray, density: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Computes the structure factor based on the radial distribution function (RDF).

    The structure factor :math:`S(q)` based on the RDF :math:`g(r)` is given by

    .. math::
        S(q) = 1 + 4 \pi \rho \int_0^\infty \mathrm{d}r r
                         \frac{\sin(qr)}{q} (g(r) - 1)\,

    where :math:`q` is the magnitude of the scattering vector. The calculation is
    performed via a discrete sine transform as implemented in :func:`scipy.fftpack.dst`.

    For an `example` take a look at :ref:`howto-saxs`.

    Parameters
    ----------
    rdf : numpy.ndarray
        radial distribution function
    r : numpy.ndarray
        equally spaced distance array on which rdf is defined
    density : float
        number density of particles

    Returns
    -------
    q : numpy.ndarray
        array of q points
    struct_factor : numpy.ndarray
        structure factor

    """
    dr = r[1] - r[0]
    q = np.pi / r[-1] * np.arange(1, len(r) + 1)
    struct_factor = 1 + 4 * np.pi * density * 0.5 * dst((rdf - 1) * r) / q * dr

    return q, struct_factor
