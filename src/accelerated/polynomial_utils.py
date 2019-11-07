#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:21:32 2019

@author: emil
"""


import numba as nb
import numpy as np


@nb.njit(nb.float64[::1](nb.float64[::1], nb.float64[::1]), fastmath = True)
def polyval(poly,x):
    # return x.reshape(-1,1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64))
    return np.sum(poly.reshape(-1,1)* x.reshape(1,-1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64)).reshape(-1,1), axis = 0)



@nb.njit(nb.float64[::1](nb.float64[::1]),
          fastmath = True, cache = True)
def polydiff(poly):
    return poly[:-1] * (poly.shape[0] - np.arange(poly.shape[0])[1:])

@nb.njit(nb.float64[:,::1](nb.float64[:,::1]),
          fastmath = True, cache = True)
def polydiff_2d(poly):
    return poly[:,:-1] * (poly.shape[1] - np.arange(poly.shape[1])[1:].reshape(1,-1))

@nb.njit(nb.float64[::1](nb.float64[::1], nb.int64),
          fastmath = True, cache = True)
def polydiff_d(poly, d):
    for i in range(d):
        poly = polydiff(poly)
    return poly



@nb.njit(nb.float64[:,::1](nb.float64[:,::1]), cache = True, parallel = True)
def polypow2(poly):
    out = np.empty((poly.shape[0], poly.shape[1]*2-1))
    zeros = np.zeros((poly.shape[0],poly.shape[1]-1))
    padded = np.concatenate((zeros,poly,zeros), axis = 1)
    flipped = poly[:,::-1]
    for i in nb.prange(poly.shape[1]*2-1):
        out[:,i] = np.sum(padded[:,i:i+poly.shape[1]] * flipped, axis = 1)
    return out



if __name__ == '__main__':
    a = np.array([[2.,1.],
                  [3,1],
                  [3,2]])
    print(polypow2(a))