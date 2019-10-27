#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:46:56 2019

@author: emil
"""


import numba as nb
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(1, 'accelerated')
from polynomial_utils import polyval
import time

p = 5
t_points = np.arange(100,dtype = np.float64)

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

plt.close('all')
t = time.time()
B = basepolys(t_points,p)
t = time.time() - t
print(t)
for i, polys in enumerate(B):
    for j in range(p+1):
        x = np.linspace(t_points[i+j],t_points[i+j+1], 100).reshape(-1)
        y = polyval(B[i,j],x)
        plt.plot(x,y)