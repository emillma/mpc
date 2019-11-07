#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:51:34 2019

@author: emil
"""

import numba as nb
import numpy as np


@nb.njit(nb.float64[:,::1](nb.float64[:,:,:], nb.float64[::1], nb.float64[:], nb.int64),
         parallel = True,fastmath = True, cache = True)
def get_polys_from_bases(B, c, t_points, p = 5):
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
def get_bases(t_points, p):
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
        B_n[:,:-1,1:] -= k1*t_points[:-p_i-1]*B[:-1] #multipli by -t[i]

        B_n[:,1:,:-1] -= k2*B[1:] #multiply by -x
        B_n[:,1:,1:]  += k2*t_points[1+p_i:]*B[1:] #multipli by -t[ik1]
        B = B_n
    return B