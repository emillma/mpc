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

@nb.njit(nb.float64[::1](nb.float64[:]), fastmath = True, parallel = True)
def polydiff(poly):
    return poly[:-1] * (poly.shape[0] - np.arange(poly.shape[0])[1:])

# @nb.njit(nb.float64[::1](nb.float64[::1]), fastmath = True, cache = True)
# def polypow2(poly):
#     return np.ascontiguousarray(np.convolve(poly, poly))
@nb.njit(nb.float64[::1](nb.float64[::1]), cache = True, parallel = True)
def polypow2(poly):
    out = np.empty(poly.shape[0]*2-1).astype(np.float64)
    zeros = np.zeros(poly.shape[0]-1).astype(np.float64)
    padded = np.concatenate((zeros,poly,zeros))
    flipped = poly[::-1]
    for i in nb.prange(poly.shape[0]*2-1):
        out[i] = np.sum(padded[i:i+poly.shape[0]] * flipped)
    return out

if __name__ == '__main__':
    poly = np.random.random(6)-0.5
    x = np.linspace(-2,2,10)

    print(np.allclose(polyval(poly,x),np.polyval(poly,x)))