#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:22:57 2019

@author: emil
"""

import numpy as np
import numba as nb


@nb.njit(nb.float64[::1](nb.float64[:],nb.float64[:]), fastmath = True, cache = True)
def get_polyroots(polynome, boundory):
    """
    return the roots of the polynoe within the boundory, as well as the boundory points
    """
    cast_t = np.float64
    length = len(polynome)
    p = np.empty((length,), dtype = np.complex128)
    for i in nb.prange(length):
        p[i] = nb.complex128(polynome[i])

    p = polynome.astype(np.complex128)
    if len(p.shape) != 1:
          raise ValueError("Input must be a 1d array.")

    non_zero = np.nonzero(p)[0]

    if len(non_zero) == 0:
        return np.zeros(0, dtype=cast_t)

    tz = int(len(p) - non_zero[-1] - 1)

    # pull out the coeffs selecting between possible zero pads
    p = p[int(non_zero[0]):int(non_zero[-1]) + 1]

    n = len(p)
    if n > 1:
        # construct companion matrix, ensure fortran order
        # to give to eigvals, write to upper diag and then
        # transpose.
        A = np.diag(np.ones((n - 2,), np.complex128), 1).T
        A[0, :] = -p[1:] / p[0]  # normalize
        roots = np.linalg.eigvals(A)
    else:
        roots = np.zeros(0, dtype=np.complex128)
    roots = roots[np.less_equal(np.abs(np.imag(roots)), 1e-9)]
    roots_real = np.real(roots)
    # add in additional zeros on the end if needed
    if tz > 0:
        roots_real =  np.hstack((roots_real, np.zeros(tz, dtype = cast_t)))

    roots_real = roots_real[np.logical_and(np.less(roots_real, boundory[1]-1e-9),
                                           np.greater(roots_real, boundory[0]+1e-9))]
    roots_real = np.hstack((boundory[0:1], roots_real, boundory[1:2]))
    return np.ascontiguousarray(np.sort(roots_real))

@nb.njit(nb.float64(nb.float64[::1],nb.float64[:]), fastmath = True, cache = True)
def get_polyroot_max(poly, boundory):
    """
    get the t E ]boundory[0]; boundory[1]], max poly(t)
    """
    n = poly.shape[0]
    poly_d = poly[:-1] * np.arange(n-1,0,-1).astype(np.float64)
    roots = get_polyroots(poly_d, boundory)[1:]
    potentials = (n - np.arange(n)).astype(np.float64).reshape(1,-1)
    if roots.shape[0] == 1:
        return roots[0]
    argmax = np.argmax(np.sum(roots.reshape(-1,1) ** potentials * poly.reshape(1,-1), axis = 1))
    return roots[argmax]


if __name__ == '__main__':
    poly = np.array([1,0,-3,0], dtype = np.float64)
    # print(poly.dtype)
    print(get_polyroot_max(poly, np.array([-1.,1.2])))