# distutils: language = c
# cython: language_level=3
#
# Copyright (c) 2025 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np

cimport cython
cimport numpy as np
from cython.parallel cimport prange
from libc cimport math


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision
cpdef tuple structure_factor(
        double[:,:] positions,
        double[:] dimensions,
        double qmin,
        double qmax,
        double thetamin,
        double thetamax,
        double[:] weights,
    ):
    r"""Calculates scattering vectors and corresponding structure factors.

    Use via ``from maicos.lib.math import structure_factor``

    The structure factors are calculated according to

    .. math::
        S(\boldsymbol{q}) = 
            \left [ \sum\limits_{k=1}^N w_j(q) \cos(\boldsymbol{qr}_j) \right ]^2 +
            \left [ \sum\limits_{k=1}^N w_j(q) \sin(\boldsymbol{qr}_j) \right ]^2 \,.

    where :math:`\boldsymbol{r}_j` is the positions vector of particle :math:`k`,
    :math:`\boldsymbol{q}` is scattering vector and the :math:`w_j` are optional
    weights. The possible scattering vectors are determined by the given cell
    ``dimensions``.

    Results are returned as arrays with three dimensions, where the index of each
    dimensions referers to the Miller indices :math:`hkl`. Based on the Miller indices
    and the returned length of the scattering vector the actual scattering vector can be
    obtained by

    .. math::
        q_{hkl} = \vert \boldsymbol{q} \vert \frac{2\pi}{L_{hkl}}

    where :math:`\vert \boldsymbol{q} \vert` are the returned lengths of the scattering vector and 
    :math:`L_{hkl}` are the components of the simulation cell.

    Parameters
    ----------
    positions : numpy.ndarray
        position array.
    dimensions : numpy.ndarray
        dimensions of the cell.
    qmin : float
        Starting scattering vector length (1/Å).
    qmax : float
        Ending scattering vector length (1/Å).
    thetamin : float
        Minimal angle (°) between the scattering vectors and the z-axis.
    thetamax : float
        Maximal angle (°) between the scattering vectors and the z-axis.
    weights : numpy.ndarray
        Atomic quantity whose :math:`S(\vert q \vert)` we are computing. Provide an
        array of ``1`` that has the same size as the postions, h.e
        ``np.ones(len(positions))``, for the standard structure factor.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        The length of the scattering vectors and the corresponding structure factors.
    """

    assert(dimensions.shape[0] == 3)
    assert(positions.shape[1] == 3)
    assert(len(weights) == len(positions))

    cdef Py_ssize_t i, h, k, l, j, n_atoms
    cdef int[::1] maxn = np.empty(3,dtype=np.int32)
    cdef double qx, qy, qz, qrr, qdotr, sin, cos, theta
    cdef double[::1] q_factor = np.empty(3,dtype=np.double)

    n_atoms = positions.shape[0]
    for i in range(3):
        q_factor[i] = 2 * np.pi / dimensions[i]
        maxn[i] = <int>math.ceil(qmax / <float>q_factor[i])

    cdef double[:,:,::1] scattering_vectors = np.zeros(maxn, dtype=np.double)
    cdef double[:,:,::1] structure_factors = np.zeros(maxn, dtype=np.double)

    for h in prange(<int>maxn[0], nogil=True):
        qx = h * q_factor[0]

        for k in range(maxn[1]):
            qy = k * q_factor[1]

            for l in range(maxn[2]):
                if (h + k + l != 0):
                    qz = l * q_factor[2]
                    qrr = math.sqrt(qx * qx + qy * qy + qz * qz)
                    theta = math.acos(qz / qrr)

                    if (qrr >= qmin and qrr <= qmax and
                          theta >= thetamin and theta <= thetamax):
                        scattering_vectors[h, k, l] = qrr

                        sin = 0.0
                        cos = 0.0
                        for j in range(n_atoms):
                            qdotr = positions[j, 0] * qx + positions[j, 1] * qy + positions[j, 2] * qz
                            sin += weights[j] * math.sin(qdotr)
                            cos += weights[j] * math.cos(qdotr)

                        structure_factors[h, k, l] += sin * sin + cos * cos

    return (np.asarray(scattering_vectors), np.asarray(structure_factors))
