#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:20:20 2019

@author: emil
"""


import numpy as np
from matplotlib import pyplot as plt
from splines import get_A_b_t_augmented
plt.close('all')
for n in range(7,51,1):
    x = np.zeros(39)
    for i in range(1,5):
        t_points =  np.linspace(-1,1,n) * float(i*0.1)
        y_points = np.arange(n) + (np.random.random(n)-0.5)*0.
        start_derivatives = np.array([0.,0.,0])
        end_derivatives = np.array([0.,10.,0])

        A, b, t_augmented =get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)
        x[i-1] = np.amax(np.abs(np.linalg.inv(A)))
    plt.plot(np.log(x))
    # print(x)