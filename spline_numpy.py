#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 17:24:00 2019

@author: emil
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
import scipy
import scipy.interpolate
from matplotlib import pyplot as plt

n = 5
x_points = np.arange(n)
y_points = np.random.random(n)
# y_points[1] = 1

x_2_points = np.zeros(x_points.shape[0]+2)
x_2_points[0] = x_points[0]
x_2_points[1] = (x_points[0] + x_points[1])/2.
x_2_points[2:-2] = x_points[1:-1]
x_2_points[-2] = (x_points[-2] + x_points[-1])/2.
x_2_points[-1] = x_points[-1]


ptentials = 5-np.arange(6)
poly = np.ones(6)
poly_d1 = np.polyder(poly,1)
poly_d2 = np.polyder(poly,2)
poly_d3 = np.polyder(poly,3)
poly_d4 = np.polyder(poly,4)

shape = 6*(n+1)+2
A = np.zeros((shape,shape))
poly_0 = slice(0,6)
A[0, poly_0] = poly * x_2_points[0]**ptentials # = y[0]
A[1, poly_0] = poly * x_2_points[1]**ptentials # = 0
A[1,-2] = -1

A[2, poly_0][:-1] = poly_d1 * x_2_points[0]**ptentials[1:] # start[0]
A[3, poly_0][:-2] = poly_d2 * x_2_points[0]**ptentials[2:] # start[1]
A[4, poly_0][:-3] = poly_d3 * x_2_points[0]**ptentials[3:] # start[2]

A[5,-2] = -1
last_poly = poly_0
for i in range(0, x_points.shape[0]):
    p_index_start = 5+6*i
    this_poly = slice(6+i*6, 6+(i+1)*6)

    A[p_index_start+0, this_poly] = poly * x_2_points[1+i]**ptentials # = y[k]
    A[p_index_start+1, this_poly] = poly * x_2_points[2+i]**ptentials # = y[k+1]

    A[p_index_start+2, this_poly][:-1] = poly_d1 * x_2_points[1+i]**ptentials[1:]
    A[p_index_start+2, last_poly][:-1] = -poly_d1 * x_2_points[1+i]**ptentials[1:]

    A[p_index_start+3, this_poly][:-2] = poly_d2 * x_2_points[1+i]**ptentials[2:] # = 0
    A[p_index_start+3, last_poly][:-2] = -poly_d2 * x_2_points[1+i]**ptentials[2:]

    A[p_index_start+4, this_poly][:-3] = poly_d3 * x_2_points[1+i]**ptentials[3:] # = 0
    A[p_index_start+4, last_poly][:-3] = -poly_d3 * x_2_points[1+i]**ptentials[3:]

    A[p_index_start+5, this_poly][:-4] = poly_d4 * x_2_points[1+i]**ptentials[4:] # = 0
    A[p_index_start+5, last_poly][:-4] = -poly_d4 * x_2_points[1+i]**ptentials[4:]

    last_poly = this_poly

poly_last = slice(-8,-2)
A[-3, poly_last][:-1] = poly_d1 * x_2_points[-1]**ptentials[1:] # end[0]
A[-2, poly_last][:-2] = poly_d2 * x_2_points[-1]**ptentials[2:] # end[1]
A[-1, poly_last][:-3] = poly_d3 * x_2_points[-1]**ptentials[3:] # end[2]

A[-9, -1] = -1
A[-14,-1] = -1




b = [y_points[0], 0, 0, 0, 0]
b += [0, y_points[1], 0, 0, 0, 0]
for i in range(x_points.shape[0]-3):
    b += [y_points[1+i], y_points[2+i], 0, 0, 0, 0]

b += [y_points[-2], 0, 0, 0, 0, 0]
b += [0, y_points[-1], 0, 0, 0, 0]
b += [0,0,0]
b = np.array(b)

solution = np.linalg.solve(A,b)
polys = solution[:-2].reshape(-1,6)

inter = scipy.interpolate.PPoly(polys.T,x_2_points)
plt.close('all')
for i in range(x_2_points.shape[0]-1):
    x = np.linspace(x_2_points[i],x_2_points[i+1], 50)
    plt.plot(x, polyval(x,polys[i,::-1]))