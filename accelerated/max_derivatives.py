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
t_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
y_points = np.random.random(n)
start_derivatives = np.array([0.,0.,0])
end_derivatives = np.array([0.,10.,0])

A, b, t_augmented = splines.get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)

A_inv = np.linalg.inv(A)
polys = (A_inv@b)[:-2].reshape(-1,6)

inter = scipy.interpolate.PPoly(polys.T,t_augmented)


plt.close('all')
for i in range(t_augmented.shape[0]-1):
    x = np.linspace(t_augmented[i],t_augmented[i+1], 50)
    plt.plot(x, np.polyval(polys[i,:],x))