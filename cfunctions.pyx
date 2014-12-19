#!python
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
import scipy.sparse as sp
import collections
from cython.parallel import parallel, prange


cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil
    void daxpy "cblas_daxpy" (int, double, const double*,
                              int, double*, int) nogil


cdef double naive_dot(double[::1] x,
                      double[::1] y,
                      int dim) nogil:

    cdef int i
    cdef double result = 0.0

    for i in range(dim):
        result += x[i] * y[i]

    return result


cpdef double run_naive_dot(double[::1] x,
                           double[::1] y,
                           int dim,
                           int number):

    cdef int i
    cdef double result

    for i in range(number):
        result = naive_dot(x, y, dim)

    return result


cpdef double run_blas_dot(double[::1] x,
                          double[::1] y,
                          int dim,
                          int number):

    cdef int i
    cdef double result

    for i in range(number):
        result = ddot(dim, &x[0], 1, &y[0], 1)

    return result
        
