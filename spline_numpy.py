#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:24:00 2019

Dont timeshift fisrt
@author: emil
"""

import numpy as np
import scipy
import scipy.interpolate
from matplotlib import pyplot as plt

def get_spline(x_points, y_points,
               start_derivatives = [0.,0.,0.], end_derivatives = [0.,0.,0.]):
    n = x_points.shape[0]

    x_2_points = np.zeros(x_points.shape[0]+2)
    x_2_points[0] = x_points[0]
    x_2_points[1] = (x_points[0] + x_points[1])/2.
    x_2_points[2:-2] = x_points[1:-1]
    x_2_points[-2] = (x_points[-2] + x_points[-1])/2.
    x_2_points[-1] = x_points[-1]


    potentials = 5-np.arange(6)
    poly = np.ones(6)
    poly_d1 = np.polyder(poly,1)
    poly_d2 = np.polyder(poly,2)
    poly_d3 = np.polyder(poly,3)
    poly_d4 = np.polyder(poly,4)

    problem_shape = 6*(n+1)+2
    A = np.zeros((problem_shape,problem_shape))
    poly_0_slice = slice(0,6)

    last_potentials = x_2_points[0]**potentials
    next_potentials = x_2_points[1]**potentials

    A[0, poly_0_slice] = poly * last_potentials # = y[0]
    A[1, poly_0_slice] = poly * next_potentials # = 0
    A[1,-2] = -1

    A[2, poly_0_slice][:-1] = poly_d1 * last_potentials[1:] # start[0]
    A[3, poly_0_slice][:-2] = poly_d2 * last_potentials[2:] # start[1]
    A[4, poly_0_slice][:-3] = poly_d3 * last_potentials[3:] # start[2]

    A[5,-2] = -1
    last_poly = poly_0_slice


    for i in range(0, x_points.shape[0]):
        p_index_start = 5+6*i
        this_poly_slice = slice(6+i*6, 6+(i+1)*6)

        #First point equal to y[i]
        A[p_index_start+0, this_poly_slice] = poly * next_potentials # = y[k]

        #Derivatives equal from last
        A[p_index_start+2, this_poly_slice][:-1] = poly_d1 * next_potentials[1:]
        A[p_index_start+2, last_poly][:-1] = -poly_d1 * next_potentials[1:]

        A[p_index_start+3, this_poly_slice][:-2] = poly_d2 * next_potentials[2:] # = 0
        A[p_index_start+3, last_poly][:-2] = -poly_d2 * next_potentials[2:]

        A[p_index_start+4, this_poly_slice][:-3] = poly_d3 * next_potentials[3:] # = 0
        A[p_index_start+4, last_poly][:-3] = -poly_d3 * next_potentials[3:]

        A[p_index_start+5, this_poly_slice][:-4] = poly_d4 * next_potentials[4:] # = 0
        A[p_index_start+5, last_poly][:-4] = -poly_d4 * next_potentials[4:]


        next_potentials = x_2_points[i+2]**potentials

        A[p_index_start+1, this_poly_slice] = poly * next_potentials # = y[k+1]

        last_poly = this_poly_slice

    poly_last = slice(-8,-2)
    A[-3, poly_last][:-1] = poly_d1 * x_2_points[-1]**potentials[1:] # end[0]
    A[-2, poly_last][:-2] = poly_d2 * x_2_points[-1]**potentials[2:] # end[1]
    A[-1, poly_last][:-3] = poly_d3 * x_2_points[-1]**potentials[3:] # end[2]

    A[-9, -1] = -1
    A[-14,-1] = -1


    b = np.zeros(problem_shape)
    b[0] = y_points[0]
    b[2:5] = start_derivatives
    b[6] = y_points[1]

    for i in range(1, x_points.shape[0]-2):
        p_index_start = 5+6*i
        b[p_index_start] = y_points[i]
        b[p_index_start+1] = y_points[i+1]

    b[5+(n-2)*6] = y_points[-2]
    b[5+(n-1)*6+1] = y_points[-1]

    solution = np.linalg.solve(A,b)
    polys = solution[:-2].reshape(-1,6)
    return polys, x_2_points

if __name__ == '__main__':
    n = 10
    x_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
    y_points = np.random.random(n)
    start_derivatives = [0.,0.,0]
    end_derivatives = [0.,0.,0]
    polys, x_points_2 = get_spline(x_points, y_points,
                                   start_derivatives, end_derivatives)

    inter = scipy.interpolate.PPoly(polys.T,x_points_2)
    plt.close('all')
    for i in range(x_points_2.shape[0]-1):
        x = np.linspace(x_points_2[i],x_points_2[i+1], 50)
        plt.plot(x, np.polyval(polys[i,:], x))