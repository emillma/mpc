#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:22:23 2019

@author: emil
"""


import numba as nb
from  roots import get_polyroots
import splines
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate
from polynomial_utils import polyval, polydiff, polypow2


@nb.njit(nb.float64[:,::1](nb.float64[::1],nb.float64[:]), fastmath = True, cache = True)
def get_max_derivatives_abs(poly, boundory):
    out = np.zeros((2,4)).astype(np.float64)
    for i in range(4):
        poly_pow = polypow2(poly)
        poly_pow_d = polydiff(poly_pow)
        roots = get_polyroots(poly_pow_d, boundory)
        roots = roots[np.where(polyval(polydiff(poly_pow_d), roots) < 0.)]
        if roots.shape[0] == 1:
            out[1,i] = roots[0]

        elif roots.shape[0] > 1:
            roots = np.sort(roots)[-2:]
            out[:,i] = roots
        poly = polydiff(poly)
    return out

#should be parallelized
@nb.njit(nb.float64[:,::1](nb.float64[:,::1], nb.float64[:]), fastmath = True, cache = True)
def get_max_t(poly, t_augmented):
    n = poly.shape[0]
    out = np.zeros((n,4)).astype(np.float64)

    out[:2] = get_max_derivatives_abs(poly[0], t_augmented[:2])
    for i in nb.prange(1,n):
        tmp = get_max_derivatives_abs(poly[i], t_augmented[i:i+2])
        out[i] = tmp[1]
        out[i-1] = np.maximum(out[i-1], tmp[0])
    return out


if __name__ == '__main__':
    n = 17
    t_points =  np.linspace(0,1,n)
    y_points = np.arange(n, dtype = np.float64)
    # y_points[-1] = 0
    start_derivatives = np.array([0.,0.,0])
    end_derivatives = np.array([0.,0.,0])

    A, b, t_augmented = splines.get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)

    A_inv = np.linalg.inv(A)
    A_inv_max = np.amax(np.abs(A_inv))
    A_inv = A_inv / A_inv_max
    polys = (A_inv@b).reshape(-1,6)
    # polys = np.linalg.solve(A,b)[:-2].reshape(-1,6)
    polys_d = np.vstack([np.polyder(i) for i in polys])

    max_t = get_max_t(polys, t_augmented)

    plt.close('all')
    fig, ax = plt.subplots(2,1, sharex = True)
    for i in range(t_augmented.shape[0]-1):
        x = np.linspace(t_augmented[i],t_augmented[i+1], 50)
        p = polys[i]

        p2 = polypow2(polys[i])
        ax[0].plot(x, np.polyval(p2,x))
        ax[0].plot(x, np.polyval(polys[i,:],x))
        ax[1].plot(x, np.polyval(polys_d[i,:],x))
        # print('h', polys[i])
        m = max_t[i]
        m0 = m[0]
        m1 = m[1]
        # ax[0].scatter(max_t[i,0], np.polyval(polys[i,:],max_t[i,0]))
        # ax[1].scatter(max_t[i,1], np.polyval(polys_d[i,:],max_t[i,1]))
        if m0:
            ax[0].scatter(m0, np.polyval(polys[i,:],m0))
        if m1:
            ax[1].scatter(m1, np.polyval(polys_d[i,:],m1))

