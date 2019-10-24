#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:20:20 2019

@author: emil
"""


import numpy as np
from matplotlib import pyplot as plt
from splines import get_A_b_t_augmented
from matplotlib import pyplot as plt

n = 31
t_points =  np.linspace(1,2,n)
# t_points = (np.random.random(n) -0.5) * np.random.random(1)*10
y_points = np.arange(n) + (np.random.random(n)-0.5)
# y_points[-1] = 0
start_derivatives = np.array([0.,0.,0])
end_derivatives = np.array([0.,0.,0])

A, b, t_augmented = get_A_b_t_augmented(t_points, y_points, start_derivatives, end_derivatives)
args = np.arange(A.shape[0])
np.random.shuffle(args)
# print(args)
start = np.empty(100)

A_inv = np.linalg.inv(A)
diff = np.amax(np.amax(np.abs(A_inv),axis = 1) - np.amin(np.abs(A_inv), axis = 1))
print(diff)
# print(np.linalg.det(A))

# plt.plot(data)