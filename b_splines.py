#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:41:13 2019

@author: emil
"""
import sys
sys.path.insert(1, 'accelerated')
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from polynomial_utils import polyval

plt.close('all')
p = 5
t_points = np.arange(3)
x = sp.symbols('x', real = True)

# def B(i, p, t, x_inter):
#     if p == 0:
#         # return 1
#         return sp.Piecewise((1, (x>=t[i]) & (x<t[i+1+p])), (0, True))
#     else:
#         return ((x-t[i])/(t[i+p] - t[i])) * B(i, p-1, t) + ((t[i+p+1] - x)/(t[i+p+1] - t[i+1])) * B(i+1, p-1, t)

def B(i, p, t, k):
    if p == 0:
        if k  >= i and k < i + 1:
            return 1
        else:
            return 0
    else:
        s = 0
        b0 = B(i, p-1, t,k)
        if b0 != 0:
            s+= b0 * ((x-t[i])/(t[i+p] - t[i]))
        b1 = B(i+1, p-1, t,k)
        if b1 != 0:
            s+= b1 * ((t[i+p+1] - x)/(t[i+p+1] - t[i+1]))
        return s

t_points = np.array([0,1,2,3,4,5,5,5,5,5,6,7,8,9,10])
t_points[5:] += 5
n = len(t_points)
t_points = np.concatenate((np.ones(p) *t_points[0],t_points,np.ones(p) *t_points[-1]))
p = 5
for i in range(n+p-1):
    for k in range(p+1):
        a = B(i,p,t_points, i+k)
        a_p = sp.Poly(a,x)
        print(a_p.all_coeffs())
        x_ = np.linspace(t_points[i+k], t_points[i+k+1], 100)
        y = polyval(np.array(a_p.all_coeffs(), dtype = np.float64), x_)
        plt.plot(x_, y)