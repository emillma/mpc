#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:24:00 2019

Dont timeshift fisrt
@author: emil
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
import scipy.interpolate
from matplotlib import pyplot as plt
import numba as nb
from numba import njit, float64, complex128, void, prange

#can be parallelized
@nb.njit(nb.float64[::1](nb.float64[:]),
         cache = True, parallel = True, fastmath = True)
def get_t_augmented(t_points):
    t_augmented = np.zeros(t_points.shape[0]+2, dtype = np.float64)
    t_augmented[0] = t_points[0]
    t_augmented[1] = (t_points[0] + t_points[1])/2.
    t_augmented[2:-2] = t_points[1:-1]
    t_augmented[-2] = (t_points[-2] + t_points[-1])/2.
    t_augmented[-1] = t_points[-1]
    return t_augmented

@nb.njit(nb.float64[::1](nb.float64[:], nb.float64[:], nb.float64[:]),
         cache = True, parallel = True, fastmath = True)
def get_b(y_points, start_derivatives, end_derivatives):
    """
    last two points are the free y parameters
    """
    n = y_points.shape[0]
    problem_shape = 6*(n+1)+2
    b = np.zeros(problem_shape)
    b[0] = y_points[0]
    b[2:5] = start_derivatives
    b[6] = y_points[1]

    for i in nb.prange(1, n-2):
        b[5+6*i] = y_points[i]
        b[5+6*i+1] = y_points[i+1]

    b[5+(n-2)*6] = y_points[-2]
    b[5+(n-1)*6+1] = y_points[-1]
    b[-3:] = end_derivatives
    return b

@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.float64[:], nb.float64[:]))(float64[:], float64[:], float64[:], float64[:]), cache = True, parallel = True, fastmath = True)
def get_A_b_t_augmented(t_points, y_points,
               start_derivatives = np.array([0.,0.,0.]),
               end_derivatives = np.array([0.,0.,0.])):
    """
    Present the problem as:
        p0[t0] = y[0]
        p0[t1] = free0 -> p0[t1] - free0 = 0
        derivatives match start Derivatives [2:5]

        match last points (-free0 for the first iteration) (-free1 for last two iterations)
        match next poit
        match last 4 Derivatives

        match end derivatives [-3:]
    """
    n = t_points.shape[0]

    t_augmented = get_t_augmented(t_points)

    potentials = 5-np.arange(6).reshape(1,-1)
    poly = np.ones(6)
    poly_d1 = np.array([5., 4., 3., 2., 1.])
    poly_d2 = np.array([ 20., 12., 6., 2])
    poly_d3 = np.array([60., 24.,  6.])
    poly_d4 = np.array([120.,  24.])

    problem_shape = 6*(n+1)+2
    A = np.zeros((problem_shape,problem_shape))


    potential_array = t_augmented.reshape(-1,1) ** potentials

    A[0, 0:6] = potential_array[0] # = y[0]
    A[1, 0:6] = potential_array[1] # = 0 subtract the free point
    A[1,-2] = -1

    A[2, 0:5] = poly_d1 * potential_array[0,1:] # start[0]
    A[3, 0:4] = poly_d2 * potential_array[0,2:] # start[1]
    A[4, 0:3] = poly_d3 * potential_array[0,3:] # start[2]

    A[5,-2] = -1

    for i in nb.prange(0, n):

        #First point equal to y[i]
        A[5+6*i+0, slice(6+i*6, 6+(i+1)*6)] = potential_array[i+1] # = y[k]
        A[5+6*i+1, 6+i*6: 6+(i+1)*6] = potential_array[2+i] # = y[k+1]

        #Derivatives equal from last
        A[7+6*i, 6+i*6: 6+(i+1)*6-1] = poly_d1 * potential_array[1+i,1:]
        A[7+6*i, 6+(i-1)*6: 6+i*6-1] = -A[5+6*i+2, 6+i*6: 6+(i+1)*6][:-1]

        A[8+6*i, 6+i*6: 6+(i+1)*6-2] = poly_d2 * potential_array[1+i,2:] # = 0
        A[8+6*i, 6+(i-1)*6: 6+i*6-2] = -A[5+6*i+3, 6+i*6: 6+(i+1)*6][:-2]

        A[9+6*i, 6+i*6: 6+(i+1)*6-3] = poly_d3 * potential_array[1+i,3:] # = 0
        A[9+6*i, 6+(i-1)*6: 6+i*6-3] = -A[5+6*i+4, 6+i*6: 6+(i+1)*6][:-3]

        A[10+6*i, 6+i*6: 6+(i+1)*6-4] = poly_d4 * potential_array[1+i,4:] # = 0
        A[10+6*i, 6+(i-1)*6: 6+(i)*6-4] = -A[5+6*i+5, 6+i*6: 6+(i+1)*6][:-4]

    A[-3, -8:-3] = poly_d1 * potential_array[-1,1:] # end[0]
    A[-2, -8:-4] = poly_d2 * potential_array[-1,2:] # end[1]
    A[-1, -8:-5] = poly_d3 * potential_array[-1,3:] # end[2]

    A[-9, -1] = -1
    A[-14,-1] = -1

    b = get_b(y_points, start_derivatives, end_derivatives)

    # solution = np.linalg.solve(A,b)
    # polys = solution[:-2].reshape(-1,6)
    return A, b, t_augmented


if __name__ == '__main__':
    n = 10
    t_points = np.arange(n)+((np.random.random(n)-0.5)*.4)

    y_points = np.random.random(n)
    start_derivatives = np.array([-10.,0.,0])
    end_derivatives = np.array([-10.,10.,0])


    A, b, t_augmented = get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)
    solution = np.linalg.solve(A,b)
    polys = solution[:-2].reshape(-1,6)

    inter = scipy.interpolate.PPoly(polys.T,t_augmented)


    plt.close('all')
    for i in range(t_augmented.shape[0]-1):
        x = np.linspace(t_augmented[i],t_augmented[i+1], 50)
        plt.plot(x, np.polyval(polys[i,:],x))