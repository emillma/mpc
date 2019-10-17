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

    # potentials = 5-np.arange(6)
    # poly = np.ones(6)
    # poly_d1 = np.array([5., 4., 3., 2., 1.])
    # poly_d2 = np.array([ 20., 12., 6., 2])
    # poly_d3 = np.array([60., 24.,  6.])
    # poly_d4 = np.array([120.,  24.])

# @nb.njit(nb.types.Tuple((nb.float64[:,:], nb.float64[:], nb.float64[:]))(float64[:], float64[:], float64[:], float64[:]), cache = True, fastmath = True, parallel = True)
def get_spline_A_b(x_points, y_points,
               start_derivatives = np.array([0.,0.,0.]), end_derivatives = np.array([0.,0.,0.])):

    n = len(x_points)

    x_augmented = np.zeros(x_points.shape[0]+2, dtype = np.float64)
    x_augmented[0] = x_points[0]
    x_augmented[1] = (x_points[0] + x_points[1])/2.
    x_augmented[2:-2] = x_points[1:-1]
    x_augmented[-2] = (x_points[-2] + x_points[-1])/2.
    x_augmented[-1] = x_points[-1]


    potentials = 5-np.arange(6)
    poly = np.ones(6)
    poly_d1 = np.array([5., 4., 3., 2., 1.])
    poly_d2 = np.array([ 20., 12., 6., 2])
    poly_d3 = np.array([60., 24.,  6.])
    poly_d4 = np.array([120.,  24.])

    problem_shape = 6*(n+1)+2
    A = np.zeros((problem_shape,problem_shape))

    potential_array = np.ones((len(x_augmented), 6), dtype =np.float64)

    for i in nb.prange(len(x_augmented)):
        potential_array[i] = x_augmented[i] ** potentials

    potential_array
#     potential_array = np.array([[1.27321204e-05, 1.21316700e-04, 1.15595370e-03, 1.10143860e-02,
#   1.04949445e-01, 1.00000000e+00]
# , [1.05196365e-01, 1.65044322e-01, 2.58940773e-01, 4.06256473e-01
# , 6.37382517e-01, 1.00000000e+00]
# , [2.19072075e+00, 1.87270606e+00, 1.60085580e+00, 1.36846851e+00
#   , 1.16981559e+00, 1.00000000e+00]
# , [1.01898751e+01, 6.40523529e+00, 4.02625536e+00, 2.53085663e+00
#   , 1.59086663e+00, 1.00000000e+00]
# , [3.29648440e+01, 1.63847878e+01, 8.14386592e+00, 4.04781271e+00
# ,  2.01191767e+00, 1.00000000e+00]])
    print('array', potential_array)

    A[0, 0:6] = x_augmented[0] ** potentials # = y[0]
    A[1, 0:6] = potential_array[1] # = 0
    A[1,-2] = -1

    A[2, 0:5] = poly_d1 * potential_array[0,1:] # start[0]
    A[3, 0:4] = poly_d2 * potential_array[0,2:] # start[1]
    A[4, 0:3] = poly_d3 * potential_array[0,3:] # start[2]

    A[5,-2] = -1


    for i in nb.prange(0, x_points.shape[0]):

        #First point equal to y[i]
        A[5+6*i+0, slice(6+i*6, 6+(i+1)*6)] = potential_array[i+1] # = y[k]
        A[5+6*i+1, 6+i*6: 6+(i+1)*6] = potential_array[2+i] # = y[k+1]

        #Derivatives equal from last
        A[5+6*i+2, 6+i*6: 6+(i+1)*6-1] = poly_d1 * potential_array[1+i,1:]
        A[5+6*i+2, 6+(i-1)*6: 6+i*6-1] = -A[5+6*i+2, 6+i*6: 6+(i+1)*6][:-1]

        A[5+6*i+3, 6+i*6: 6+(i+1)*6-2] = poly_d2 * potential_array[1+i,2:] # = 0
        A[5+6*i+3, 6+(i-1)*6: 6+i*6-2] = -A[5+6*i+3, 6+i*6: 6+(i+1)*6][:-2]

        A[5+6*i+4, 6+i*6: 6+(i+1)*6-3] = poly_d3 * potential_array[1+i,3:] # = 0
        A[5+6*i+4, 6+(i-1)*6: 6+i*6-3] = -A[5+6*i+4, 6+i*6: 6+(i+1)*6][:-3]

        A[5+6*i+5, 6+i*6: 6+(i+1)*6-4] = poly_d4 * potential_array[1+i,4:] # = 0
        A[5+6*i+5, 6+(i-1)*6: 6+(i)*6-4] = -A[5+6*i+5, 6+i*6: 6+(i+1)*6][:-4]

    A[-3, -8:-3] = poly_d1 * potential_array[-1,1:] # end[0]
    A[-2, -8:-4] = poly_d2 * potential_array[-1,2:] # end[1]
    A[-1, -8:-5] = poly_d3 * potential_array[-1,3:] # end[2]

    A[-9, -1] = -1
    A[-14,-1] = -1

    b = np.zeros(problem_shape)
    b[0] = y_points[0]
    b[2:5] = start_derivatives
    b[6] = y_points[1]

    for i in nb.prange(1, x_points.shape[0]-2):
        b[5+6*i] = y_points[i]
        b[5+6*i+1] = y_points[i+1]

    b[5+(n-2)*6] = y_points[-2]
    b[5+(n-1)*6+1] = y_points[-1]

    # solution = np.linalg.solve(A,b)
    # polys = solution[:-2].reshape(-1,6)
    return A, b, x_augmented


if __name__ == '__main__':
    n = 3
    x_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
    y_points = np.random.random(n)
    start_derivatives = np.array([0.,0.,0])
    end_derivatives = np.array([0.,0.,0])
    polys = np.empty((0,0))
    # for i in range(100):
    A, b, x_augmented = get_spline_A_b(x_points, y_points, start_derivatives, end_derivatives)
    solution = np.linalg.solve(A,b)
    polys = solution[:-2].reshape(-1,6)


    inter = scipy.interpolate.PPoly(polys.T,x_augmented)


    plt.close('all')
    for i in range(x_augmented.shape[0]-1):
        x = np.linspace(x_augmented[i],x_augmented[i+1], 50)
        plt.plot(x, np.polyval(polys[i,:],x))