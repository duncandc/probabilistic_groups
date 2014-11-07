# cython: profile=True
# filename: build_perfect_groups.pyx
# cython: boundscheck=False, overflowcheck=False

import numpy as np
import scipy.sparse

cimport numpy as np
cimport libc.stdlib as stdlib
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_group_matrix(IDs):
    
    cdef np.ndarray[np.intp_t, ndim=1] cIDs = np.ascontiguousarray(IDs,dtype=np.int)
    cdef int N = len(IDs)
    cdef int i,j, 
    cdef np.intp_t outer
    
    results = coo_entries()
    
    for i in range(0,N):
        outer = cIDs[i]
        for j in range(i+1,N):
            if outer==cIDs[j]: results.add(i,j,1)
    
    return results.to_matrix(shape=(N, N))

@cython.boundscheck(False)
@cython.wraparound(False)
def get_group_IDs(host_ID, halo_ID):
    
    cdef int N = len(halo_ID)
    cdef np.ndarray[np.intp_t, ndim=1] group_ID = np.zeros((N,), dtype=np.int)
    
    group_ID = host_ID #satellites
    
    for i in range(0,N):
        if host_ID[i]==-1: group_ID[i]=halo_ID[i] #centrals

    return group_ID


# Utility for building a coo matrix incrementally
cdef class coo_entries:
    cdef:
        np.intp_t n, n_max
        np.ndarray i, j
        np.ndarray v
        np.intp_t *i_data
        np.intp_t *j_data
        np.float64_t *v_data
    
    def __init__(self):
        self.n = 0
        self.n_max = 10
        self.i = np.empty(self.n_max, dtype=np.intp)
        self.j = np.empty(self.n_max, dtype=np.intp)
        self.v = np.empty(self.n_max, dtype=np.float64)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)

    cdef void add(coo_entries self, np.intp_t i, np.intp_t j, np.float64_t v):
        cdef np.intp_t k
        if self.n == self.n_max:
            self.n_max *= 2
            self.i.resize(self.n_max)
            self.j.resize(self.n_max)
            self.v.resize(self.n_max)
            self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
            self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
            self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        k = self.n
        self.i_data[k] = i
        self.j_data[k] = j
        self.v_data[k] = v
        self.n += 1

    def to_matrix(coo_entries self, shape=None):
        # Shrink arrays to size
        self.i.resize(self.n)
        self.j.resize(self.n)
        self.v.resize(self.n)
        self.i_data = <np.intp_t *>np.PyArray_DATA(self.i)
        self.j_data = <np.intp_t *>np.PyArray_DATA(self.j)
        self.v_data = <np.float64_t*>np.PyArray_DATA(self.v)
        self.n_max = self.n
        return scipy.sparse.coo_matrix((self.v, (self.i, self.j)), shape=shape, dtype=np.float64)
        

