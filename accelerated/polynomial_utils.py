#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:21:32 2019

@author: emil
"""


import numba as nb
import numpy as np


@nb.njit(nb.float64[::1](nb.float64[::1], nb.float64[::1]), fastmath = True, parallel = True)
def polyval(poly,x):
    # return x.reshape(-1,1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64))
    return np.sum(poly.reshape(-1,1)* x.reshape(1,-1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64)).reshape(-1,1), axis = 0)



@nb.njit(nb.float64[::1](nb.float64[::1]),
          fastmath = True, parallel = True, cache = True)
def polydiff(poly):
    return poly[:-1] * (poly.shape[0] - np.arange(poly.shape[0])[1:])

@nb.njit(nb.float64[::1](nb.float64[::1], nb.int64),
          fastmath = True, cache = True)
def polydiff_d(poly, d):
    for i in range(d):
        poly = polydiff(poly)
    return poly



@nb.njit(nb.float64[::1](nb.float64[::1]), cache = True, parallel = True)
def polypow2(poly):
    out = np.empty(poly.shape[0]*2-1).astype(np.float64)
    zeros = np.zeros(poly.shape[0]-1).astype(np.float64)
    padded = np.concatenate((zeros,poly,zeros))
    flipped = poly[::-1]
    for i in nb.prange(poly.shape[0]*2-1):
        out[i] = np.sum(padded[i:i+poly.shape[0]] * flipped)
    return out

@nb.njit(nb.float64[:,::1](nb.float64[:,:,:], nb.float64[::1], nb.float64[:], nb.int64),
         parallel = True,fastmath = True, cache = True)
def B2polys(B, c, t_points, p = 5):
    n = B.shape[0]
    B = B * c.reshape(-1,1,1)
    args1 = (np.arange(n).reshape(-1,1) + np.arange(p+1).reshape(1,-1))[:-p]
    args2 = p-np.arange(p+1)
    s = np.empty((n-p,p+1,p+1))
    for i in nb.prange(n-p):
        for j in range(p+1):
            s[i,j] = B[args1[i,j], args2[j]]
    s = np.sum(s, axis = 1)
    return s

@nb.njit(nb.float64[:,:,::1](nb.float64[::1], nb.int64),
         cache = True, fastmath = True)
def basepolys(t_points, p):
    t_points = t_points.reshape(-1,1,1)
    n = t_points.shape[0] -1
    B = np.zeros((n,1,1)).astype(np.float64)
    B[:,0,-1] = 1
    for p_i in range(1,p+1):
        B_n = np.zeros((B.shape[0]-1,B.shape[1]+1, B.shape[2]+1))

        k1 = 1/(t_points[p_i:] - t_points[:-p_i])
        k2 = k1[1:]
        k1 = k1[:-1]
        B_n[:,:-1,:-1] += k1*B[:-1] #multiply by x
        B_n[:,:-1,1:] -= k1*t_points[:-p_i-1]*B[:-1] #multipli by -ti

        B_n[:,1:,:-1] -= k2*B[1:] #multiply by -x
        B_n[:,1:,1:]  += k2*t_points[1+p_i:]*B[1:] #multipli by -t(ik1)
        B = B_n
    return B

if __name__ == '__main__':
    poly = np.random.random(6)-0.5
    x = np.linspace(-2,2,10)

    print(np.allclose(polyval(poly,x),np.polyval(poly,x)))