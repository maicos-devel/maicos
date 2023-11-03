# distutils: language = c
# cython: language_level=3
#
# Copyright (c) 2023 Authors and contributors
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
cpdef tuple compute_structure_factor(
        double[:,:] positions,
        double[:] dimensions,
        double start_q,
        double end_q,
        double mintheta,
        double maxtheta,
        double[:] weights,
    ):
    r"""Calculates :math:`S(\vert q \vert)` for all possible :math:`q` values.

    Returns the :math:`q` values as well as the structure factor.

    Use via ``from maicos.lib.math import compute_structure_factor``

    Parameters
    ----------
    positions : numpy.ndarray
        position array.
    dimensions : numpy.ndarray
        dimensions of the cell.
    startq : float
        Starting q (1/Å).
    endq : float
        Ending q (1/Å).
    mintheta : float
        Minimal angle (°) between the q vectors and the z-axis.
    maxtheta : float
        Maximal angle (°) between the q vectors and the z-axis.
    weights : numpy.ndarray
        Atomic quantity whose :math:`S(\vert q \vert)` we are computing. Provide an
        array of ``1`` that has the same size as the postions, i.e
        ``np.ones(len(positions))``, for the standard structure factor

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        The q values and the corresponding structure factor.
    """

    assert(dimensions.shape[0] == 3)
    assert(positions.shape[1] == 3)
    assert(len(weights) == len(positions))

    cdef Py_ssize_t i, j, k, l, n_atoms
    cdef int[::1] maxn = np.empty(3,dtype=np.int32)
    cdef double qx, qy, qz, qrr, qdotr, sin, cos, theta
    cdef double[::1] q_factor = np.empty(3,dtype=np.double)

    n_atoms = positions.shape[0]
    for i in range(3):
        q_factor[i] = 2 * np.pi / dimensions[i]
        maxn[i] = <int>math.ceil(end_q / <float>q_factor[i])

    cdef double[:,:,::1] S_array = np.zeros(maxn, dtype=np.double)
    cdef double[:,:,::1] q_array = np.zeros(maxn, dtype=np.double)

    for i in prange(<int>maxn[0], nogil=True):
        qx = i * q_factor[0]

        for j in range(maxn[1]):
            qy = j * q_factor[1]

            for k in range(maxn[2]):
                if (i + j + k != 0):
                    qz = k * q_factor[2]
                    qrr = math.sqrt(qx * qx + qy * qy + qz * qz)
                    theta = math.acos(qz / qrr)

                    if (qrr >= start_q and qrr <= end_q and
                          theta >= mintheta and theta <= maxtheta):
                        q_array[i, j, k] = qrr

                        sin = 0.0
                        cos = 0.0
                        for l in range(n_atoms):
                            qdotr = positions[l, 0] * qx + positions[l, 1] * qy + positions[l, 2] * qz
                            sin += weights[l] * math.sin(qdotr)
                            cos += weights[l] * math.cos(qdotr)

                        S_array[i, j, k] += sin * sin + cos * cos

    return (np.asarray(q_array), np.asarray(S_array))
