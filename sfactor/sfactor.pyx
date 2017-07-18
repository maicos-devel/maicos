#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

#Import python packages
import numpy as np
#Import C derivates for speed up
cimport numpy as np
cimport cython
from libc cimport math
from cython.parallel cimport prange

cdef double[:] CMFP_H = np.array((0.493,0.323,0.14,0.041,10.511,26.126,3.142,57.8,0.003),dtype=np.double);

cdef void distance_matrix( double[:,:] positions, int n_atoms,
                                  double[:] boxdimensions,
                                  double[:,:] dist_mat) nogil:
  """Calculates the upper triangle of the Euclidean distance matrix
  using the minimum image convention for rectengular box."""

  cdef int i, j;
  cdef double xr, yr, zr;

  for i in prange(<int>n_atoms,nogil=True):

    for j in range(n_atoms):
      if (j > i):
        xr = positions[i,0] - positions[j,0];
        yr = positions[i,1] - positions[j,1];
        zr = positions[i,2] - positions[j,2];

        # minimum image
        xr = xr - boxdimensions[0]*math.round(xr/boxdimensions[0]);
        yr = yr - boxdimensions[1]*math.round(yr/boxdimensions[1]);
        zr = zr - boxdimensions[2]*math.round(zr/boxdimensions[2]);

        dist_mat[i,j] = math.sqrt(xr*xr+yr*yr+zr*zr);

cdef inline double CMSF(double q, int nh, double[:] CMFP) nogil:
  """Calculates the form factor for the given Cromer-Mann scattering parameters."""

  cdef double q2, form_factor;
  cdef Py_ssize_t i;

  if nh > 0:
      form_factor = CMSF(q,0,CMFP) + nh*CMSF(q,0,CMFP_H)
  else:
      form_factor = CMFP[8]
      q2 = q*q/(8*math.pi*math.pi*10*10) # factor of 10 to convert from 1/nm to 1/Angstroms
      for i in range(4):
          form_factor =  form_factor + CMFP[i] * math.exp(-1*CMFP[i+4]*q2)

  return form_factor

cdef double debye(double qrr, int[:] indices, double[:,:] dist_mat,
                          int n_atoms, int[:] nh,
                          double[:,:] CMFP, double[:] form_factors) nogil:
  """Calculates the scattering intensity according to the Debye formula
  for the given Cromer-Mann scattering parameters."""

  cdef double qr, scat_int = 0
  cdef int i, j

  #calculate form factors for given qrr
  for i in range(<int>CMFP.shape[0]):
    form_factors[i] = CMSF(qrr, nh[i], CMFP[i,:])

  for i in range(n_atoms):

    for j in range(n_atoms):
      if (j > i):
        qr = qrr*dist_mat[i,j]
        scat_int += 2*math.sin(qr)/qr*form_factors[indices[i]]*form_factors[indices[j]]

  return scat_int

cpdef tuple compute_scattering_intensity(double[:,:] positions, int n_atoms,
                        int[:] indices, double [:,:] CMFP, int[:] nh,
                        double[:] boxdimensions,
                        double start_q, double end_q):
    """Calculates S(|q|) for all possible q values. Returns the q values as well as the scattering factor."""

    assert(boxdimensions.shape[0]==3);
    assert(positions.shape[0]==n_atoms);
    assert(positions.shape[1]==3);

    cdef int i, j, k;
    cdef double qx, qy, qr, qz, qrr;

    cdef int[:]        maxn = np.empty(3, dtype=np.int32);
    cdef double[:] q_factor = np.empty(3, dtype=np.double);

    for i in range(3):
        q_factor[i] = 2*math.pi/boxdimensions[i];
        maxn[i] = <int>math.ceil(end_q/<float>q_factor[i]);

    cdef double[:,:]  dist_mat  = np.zeros((n_atoms, n_atoms), dtype=np.double);
    cdef double[:] form_factors = np.zeros(len(CMFP), dtype=np.double);
    cdef double[:,:,:] q_array  = np.zeros(maxn, dtype=np.double);
    cdef double[:,:,:] S_array  = np.zeros(maxn, dtype=np.double);

    distance_matrix(positions, n_atoms, boxdimensions, dist_mat);

    with nogil:
      for i in range(maxn[0]):
        qx = i * q_factor[0];

        for j in range(maxn[1]):
            qy = j * q_factor[1];

            for k in range(maxn[2]):
                if (i + j + k != 0):
                    qz = k * q_factor[2];
                    qrr = math.sqrt(qx*qx+qy*qy+qz*qz);

                    if (qrr >= start_q and qrr <= end_q):
                        q_array[i,j,k] = qrr;
                        debye(qrr, indices, dist_mat, n_atoms, nh, CMFP, form_factors)
                        S_array[i,j,k] = <double>

    return (q_array,S_array);
