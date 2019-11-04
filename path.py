#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:11:08 2019

@author: emil
"""


import numpy as np
from matplotlib import pyplot as plt
if 0:
    from accelerated.polynomial_utils import get_bases, polyval, get_polys_from_bases, polydiff_d, polydiff

gates_n = 0
p = 5
n = p + (p-1) + 6*gates_n

lbda_points = (np.ones(n + 2*p,dtype = np.float64))
lbda_points = np.cumsum(lbda_points) - 1 - p
bases = get_bases(lbda_points, p)

c = np.arange(bases.shape[0])

start   = np.array([1,0,0,0], dtype = np.float64)
end     = np.array([2,0,0,0], dtype = np.float64)
c = np.zeros(bases.shape[0], dtype = np.float64)
# c[4] = 10


problem = np.zeros((4,4), dtype = np. float64)
for i in range(4):
    for degree in range(4):
        problem[degree,i] = polyval(polydiff_d(bases[i,5-i], degree), lbda_points[p:p+1])

gate_vars = np.zeros(4, dtype = np.float64)
for i in range(4):
    gate_vars[i] = polyval(polydiff_d(bases[p-1,0], i), lbda_points[p:p+1])


A = np.linalg.inv(problem)
c0 = A@(np.array([1.,0,0,0])-gate_vars*c[p-1:p])
c[:4] = c0


problem = np.zeros((4,4), dtype = np. float64)
for i in range(4):
    for degree in range(4):
        problem[degree,i] = polyval(polydiff_d(bases[i+bases.shape[0]-4,4-i], degree), lbda_points[-p-1:-p])

gate_vars = np.zeros(4, dtype = np.float64)
for i in range(4):
    gate_vars[i] = polyval(polydiff_d(bases[-p,-1], i), lbda_points[-p-1:-p])


A = np.linalg.inv(problem)
c0 = A@(np.array([1,0,0,0])-gate_vars*c[-p:-p+1])
c[-4:] = c0


s = get_polys_from_bases(bases, c, lbda_points, p)

plt.close('all')
fig, ax = plt.subplots(3, 1, sharex = True)

for t_i, poly in enumerate(s):
        t0 = lbda_points[t_i + p]
        t1 = lbda_points[t_i + p + 1]
        x_ = np.linspace(t0, t1, 100)
        print(t0)
        ax[0].plot(x_,polyval(poly, x_))



# for t_i,p_tmp in enumerate(bases * c[:,None,None]):
for t_i,p_tmp in enumerate(bases ):
        for t_j in range(0,p+1):

                poly = p_tmp[t_j]
                t0 = lbda_points[t_i + t_j]
                t1 = lbda_points[t_i + t_j + 1]

                x_ = np.linspace(t0, t1, 100)
                y1 = polyval(poly, x_)

                ax[1].plot(x_, y1)
                for d in range(2):
                    p_d = polydiff_d(poly, d)
                    y = polyval(p_d, x_)
                    ax[1+d].plot(x_, y)










