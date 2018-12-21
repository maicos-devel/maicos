# distutils: language = c
# cython: language_level=3

import numpy as np

cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef tuple compute_structure_factor(double[:,:] positions, double[:] boxdimensions,
                                     double start_q, double end_q, double mintheta,
                                     double maxtheta):
    """Calculates S(|q|) for all possible q values. Returns the q values as well as the scattering factor."""

    assert(boxdimensions.shape[0]==3)
    assert(positions.shape[1]==3)

    cdef Py_ssize_t i, j, k, l, n_atoms
    cdef int[:] maxn = np.empty(3,dtype=np.int32)
    cdef double qx, qy, qz, qrr, qdotr, sin, cos, theta
    cdef double[:] q_factor = np.empty(3,dtype=np.double)

    n_atoms = positions.shape[0]
    for i in range(3):
        q_factor[i] = 2*np.pi/boxdimensions[i]
        maxn[i] = <int>math.ceil(end_q/<float>q_factor[i])

    cdef double[:,:,:] S_array = np.zeros(maxn, dtype=np.double)
    cdef double[:,:,:] q_array = np.zeros(maxn, dtype=np.double)

    for i in prange(<int>maxn[0],nogil=True):
        qx = i * q_factor[0]

        for j in range(maxn[1]):
            qy = j * q_factor[1]

            for k in range(maxn[2]):
                if (i + j + k != 0):
                    qz = k * q_factor[2]
                    qrr = math.sqrt(qx*qx+qy*qy+qz*qz)
                    theta = math.acos(qz / qrr)

                    if (qrr >= start_q and qrr <= end_q and
                          theta >= mintheta and theta <= maxtheta):
                        q_array[i,j,k] = qrr

                        sin = 0
                        cos = 0
                        for l in range(n_atoms):
                            qdotr = positions[l,0]*qx + positions[l,1]*qy + positions[l,2]*qz
                            sin += math.sin(qdotr)
                            cos += math.cos(qdotr)

                        S_array[i,j,k] += sin*sin + cos*cos

    return (q_array,S_array)
