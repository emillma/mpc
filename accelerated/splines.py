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



@nb.njit(nb.types.Tuple((nb.float64[:,:], nb.float64[:]))(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]), cache = True, parallel = True, fastmath = True)
def get_spline(x_points, y_points,
               start_derivatives,
               end_derivatives):

    x_2_points = np.zeros(len(x_points)+2)
    x_2_points[0] = x_points[0]
    x_2_points[1] = (x_points[0] + x_points[1])/2.
    x_2_points[2:-2] = x_points[1:-1]
    x_2_points[-2] = (x_points[-2] + x_points[-1])/2.
    x_2_points[-1] = x_points[-1]


    potentials = 5-np.arange(6)
    poly = np.ones(6)
    poly_d1 = np.array([5., 4., 3., 2., 1.])
    poly_d2 = np.array([ 20., 12., 6., 2])
    poly_d3 = np.array([60., 24.,  6.])
    poly_d4 = np.array([120.,  24.])

    problem_shape = 6*(len(x_points)+1)+2
    A = np.zeros((problem_shape,problem_shape))

    potentials_matrix = np.zeros((len(x_points_2), len(potentials)), dtype = np.float64)
    for i in nb.prange(len(x_points_2)):
        potentials_matrix[i,:] = x_points_2[i] ** potentials


    A[0, 0: 6] = potentials_matrix[0] # = y[0]
    A[1, 0: 6] = potentials_matrix[1] # = 0
    A[1,-2] = -1

    A[2, 0: 6][:-1] = poly_d1 * potentials_matrix[0][1:] # start[0]
    A[3, 0: 6][:-2] = poly_d2 * potentials_matrix[0][2:] # start[1]
    A[4, 0: 6][:-3] = poly_d3 * potentials_matrix[0][3:] # start[2]

    A[5,-2] = -1


    for i in nb.prange(0, x_points.shape[0]):
        p_index_start = 5+6*i

        #First point equal to y[i]
        A[p_index_start+0, 6+i*6: 6+(i+1)*6] = potentials_matrix[i+1] # = y[k]
        A[p_index_start+1, 6+i*6: 6+(i+1)*6] = potentials_matrix[i+2] # = y[k+1]

        #Derivatives equal from last
        A[p_index_start+2, 6+i*6: 6+(i+1)*6][:-1] = poly_d1 * potentials_matrix[i+1][1:]
        A[p_index_start+2, 6+(i-1)*6: 6+i*6][:-1] = -poly_d1 * potentials_matrix[i+1][1:]

        A[p_index_start+3, 6+i*6: 6+(i+1)*6][:-2] = poly_d2 * potentials_matrix[i+1][2:] # = 0
        A[p_index_start+3, 6+(i-1)*6: 6+i*6][:-2] = -poly_d2 * potentials_matrix[i+1][2:]

        A[p_index_start+4, 6+i*6: 6+(i+1)*6][:-3] = poly_d3 * potentials_matrix[i+1][3:] # = 0
        A[p_index_start+4, 6+(i-1)*6: 6+i*6][:-3] = -poly_d3 * potentials_matrix[i+1][3:]

        A[p_index_start+5, 6+i*6: 6+(i+1)*6][:-4] = poly_d4 * potentials_matrix[i+1][4:] # = 0
        A[p_index_start+5, 6+(i-1)*6: 6+(i)*6][:-4] = -poly_d4 * potentials_matrix[i+1][4:]



    A[-3, -8: -3] = poly_d1 * x_2_points[-1]**potentials[1:] # end[0]
    A[-2, -8: -4] = poly_d2 * x_2_points[-1]**potentials[2:] # end[1]
    A[-1, -8: -5] = poly_d3 * x_2_points[-1]**potentials[3:] # end[2]

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

    solution = np.linalg.solve(A,b)
    polys = solution[:-2].reshape(-1,6)
    # polys = np.empty((0,0), dtype = np.float64)
    # x_2_points = np.empty((0,), dtype = np.float64)
    return polys, x_2_points


if __name__ == '__main__':
    n = 5
    x_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
    y_points = np.random.random(n)
    start_derivatives = np.array([0.,0.,0])
    end_derivatives = np.array([0.,0.,0])
    polys = np.empty((0,0))
    x_points_2 = np.empty((0,))
    for i in range(1):
        polys, x_points_2 = get_spline(x_points, y_points, start_derivatives, end_derivatives)

    inter = scipy.interpolate.PPoly(polys.T,x_points_2)
    plt.close('all')
    for i in range(x_points_2.shape[0]-1):
        x = np.linspace(x_points_2[i],x_points_2[i+1], 50)
        plt.plot(x, np.polyval(polys[i,:],x))