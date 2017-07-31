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

cdef float[:] CMFP_H = np.array((0.493,0.323,0.14,0.041,10.511,26.126,3.142,57.8,0.003),dtype=np.float32);

cdef inline int Trag_noeq(int row, int col, int N) nogil:
  """Returns the index of a 1D array representation from a upper triangular
  matrix without diagonal with size N. Taken from:
  http://www.codeguru.com/cpp/cpp/algorithms/general/article.php/c11211/TIP-Half-Size-Triangular-Matrix.htm"""

  return row*(N-1) - (row-1)*((row-1) + 1)/2 + col - row - 1;

cdef void distance_matrix( float[:,:] positions, int n_atoms,
                                  float[:] boxdimensions,
                                  float[:] dist_mat, int nt) nogil:
  """Calculates the upper triangle of the Euclidean distance matrix
  using the minimum image convention for rectengular box."""

  cdef int i, j;
  cdef float xr, yr, zr;

  for i in prange(n_atoms, schedule="dynamic", num_threads=nt):

    for j in range(i+1,n_atoms):
        xr = positions[i,0] - positions[j,0];
        yr = positions[i,1] - positions[j,1];
        zr = positions[i,2] - positions[j,2];

        # minimum image
        xr = xr - boxdimensions[0]*math.round(xr/boxdimensions[0]);
        yr = yr - boxdimensions[1]*math.round(yr/boxdimensions[1]);
        zr = zr - boxdimensions[2]*math.round(zr/boxdimensions[2]);

        dist_mat[Trag_noeq(i, j, n_atoms)] = math.sqrt(xr*xr+yr*yr+zr*zr);

cdef float CMSF(float q, int nh, float[:] CMFP) nogil:
  """Calculates the form factor for the given Cromer-Mann scattering parameters."""

  cdef float q2, form_factor;
  cdef int i;

  if nh > 0:
      form_factor = CMSF(q,0,CMFP) + nh*CMSF(q,0,CMFP_H)
  else:
      form_factor = CMFP[8]
      q2 = q*q/(8*math.pi*math.pi*10*10) # factor of 10 to convert from 1/nm to 1/Angstroms
      for i in range(4):
          form_factor =  form_factor + CMFP[i] * math.exp(-1*CMFP[i+4]*q2)

  return form_factor;

cdef double debye(float q, int[:] indices, float[:] dist_mat,
                          int n_atoms, int[:] nh,
                          float[:,:] CMFP, float[:] form_factors, float r_max) nogil:
  """Calculates the scattering intensity according to the Debye formula
  for the given Cromer-Mann scattering parameters."""

  cdef double r, norm = 0, scat_int = 0
  cdef int i, j

  #calculate form factors for given q
  for i in range(form_factors.shape[0]):
    form_factors[i] = CMSF(q, nh[i], CMFP[i,:])

  #Debye eq: sum_i f[i]*f[i] + sum_i sum_{j>i} 2 *f _i * f_j * sin(q * r_ij) / ( q * r_ij )
  for i in range(n_atoms):
    scat_int = scat_int + form_factors[indices[i]] * form_factors[indices[i]];
    for j in range(i+1,n_atoms):
        r = dist_mat[Trag_noeq(i, j, n_atoms)];
        #damp_factor = math.sin(math.pi*r/r_max) / (math.pi*r/r_max)
        scat_int = scat_int + 2 * math.sin( q * r ) / ( q * r ) \
                             * form_factors[indices[i]] * form_factors[indices[j]];

  #normalization: sum_i f_i**2
  for i in range(n_atoms):
    norm = norm + form_factors[indices[i]]*form_factors[indices[i]]

  scat_int = scat_int/norm

  return scat_int

cpdef double[:] compute_scattering_intensity(float[:,:] positions, int n_atoms,
                        int[:] indices, float [:,:] CMFP, int[:] nh,
                        float[:] boxdimensions, float[:] dist_mat,
                        float [:] form_factors,
                        float start_q, float dq, int nbins,
                        int nt):
  """Calculates S(|q|) for all possible q values. Returns the q values as well as the scattering factor."""

  assert(boxdimensions.shape[0]==3);
  assert(positions.shape[0]==n_atoms);
  assert(positions.shape[1]==3);

  cdef int i;
  cdef float r_max, q;

  r_max = max(boxdimensions[0], boxdimensions[1]);
  r_max = max(boxdimensions[2], r_max) / 2.0;
  ref_q = math.pi / r_max;

  cdef double[:] S_array  = np.zeros(nbins, dtype=np.double);

  distance_matrix(positions, n_atoms, boxdimensions, dist_mat, nt);

  for i in prange(nbins, nogil=True, schedule="dynamic", num_threads=nt):
    q = start_q + (i + 0.5) * dq
    if q > ref_q:
        S_array[i] = debye(q, indices, dist_mat, n_atoms, nh, CMFP, form_factors, r_max);

  return S_array;
