#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:22:23 2019

@author: emil
"""


import numba as nb
import roots
import splines
import numpy as np
from matplotlib import pyplot as plt
import scipy.interpolate

n = 20
t_points = np.arange(n,dtype = np.float64)
y_points = np.random.random(n)
start_derivatives = np.array([0.,0.,0])
end_derivatives = np.array([0.,10.,0])

A, b, t_augmented = splines.get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)

A_inv = np.linalg.inv(A)
polys = (A_inv@b)[:-2].reshape(-1,6)
polys_d = np.vstack([np.polyder(i) for i in polys])



#should be parallelized
@nb.njit(nb.float64[:,::1](nb.float64[:,::1], nb.float64[::1]), fastmath = True, cache = True)
def get_max_t(poly, t_augmented):
    n = poly.shape[0]
    degree = 5
    out = poly.copy()
    multiplier = np.array([5., 4., 3., 2., 1.], dtype = np.float64)
    derivator = np.ones(degree+1).astype(np.float64)

    potentials = 5-np.arange(6).reshape(1,-1)
    max_t = np.zeros((n,4), dtype = np.float64)

    poly_tmp = poly.copy()
    poly_tmp_d = poly.copy()
    max_t_tmp = np.empty((n,6), dtype = np.float64)
    
    for i in range(1,5):
        poly_tmp_d[:,1:] = poly_tmp[:,:-1] * multiplier.reshape(1,-1)
        poly_tmp_d[:,0] = 0.

        for j in range(n):

            tmp = roots.get_polyroots(poly_tmp_d[j], t_augmented[j:j+2])[1:]
            max_t[j,i-1] = tmp[np.argmax(np.sum(tmp.reshape(-1,1) ** potentials * poly_tmp[j].reshape(1,-1), axis = 1))]

        poly_tmp = poly_tmp_d

    return max_t



max_t = get_max_t(polys, t_augmented)
# ax[0].scatter(max_t[:,0], inter.__call__(max_t[:,0]))
plt.close('all')
fig, ax = plt.subplots(2,1, sharex = True)
for i in range(t_augmented.shape[0]-1):
    x = np.linspace(t_augmented[i],t_augmented[i+1], 50)
    ax[0].plot(x, np.polyval(polys[i,:],x))
    ax[1].plot(x, np.polyval(polys_d[i,:],x))

    ax[0].scatter(max_t[i,0], np.polyval(polys[i,:],max_t[i,0]))
    ax[1].scatter(max_t[i,1], np.polyval(polys_d[i,:],max_t[i,1]))
