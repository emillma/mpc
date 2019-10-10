#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:14:05 2019

@author: emil
"""


from optimization_problem import get_problem
import numpy as np
import sympy as sp
from utils import plot_piecewise
from matplotlib import pyplot as plt
# out1, out2 = get_problem(0)
x_polys, lbda_parts, x_points, x_start, x_end, lbda = out1
lbda_polys, t_parts, lbda_points, t = out2
n = len(x_points)
x_points_vals = dict(zip(x_points,list(np.linspace(0,10, n) + np.random.random(n)*5)))
x_start_vals = dict(zip(x_start, [0.,0.,0.,0.]))
x_end_vals = dict(zip(x_end, [0.,0.,0.,0.]))

args = []
for i in range(len(x_polys)):
    args.append((sum([x_polys[i].subs(x_points_vals).subs(x_start_vals).subs(x_end_vals)[j]
                * lbda**(5-j) for j in range(6)]), lbda<lbda_parts[i+1]))
f = sp.Piecewise(*args)

plt.close('all')
fig, ax = plt.subplots(3, 2, sharex='col')
plot_piecewise(f,lbda,np.linspace(0,1,200), ax[0,0])
plot_piecewise(f.diff(lbda,1),lbda,np.linspace(0,1,200), ax[1,0])
plot_piecewise(f.diff(lbda,2),lbda,np.linspace(0,1,200), ax[1,1])
plot_piecewise(f.diff(lbda,3),lbda,np.linspace(0,1,200), ax[2,0])
plot_piecewise(f.diff(lbda,4),lbda,np.linspace(0,1,200), ax[2,1])

# ax[0,0].scatter(x,y)