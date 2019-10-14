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
if 1:
    out1, out2 = get_problem(0)
    x_polys, lbda_parts, x_points, x_start, x_end, lbda = out1
    lbda_polys, t_parts, lbda_points, t = out2
    n = len(x_points)


x_points_vals = dict(zip(x_points[1:-1],list(np.linspace(0,10, n)[1:-1] + np.random.random(n-2))))
x_start_vals = dict(zip(x_start, [0.,0.,0.,0.]))
x_start_end = dict(zip(x_end, [10.,0.,0.,0.]))
lbda_points_vals = dict(zip(lbda_points[1:-1],list(np.linspace(0,1, n)[1:-1] + np.random.random(n-2)*0.05)))

all_vals = {}
for i in [x_points_vals,x_start_vals,x_start_end,lbda_points_vals]:
    all_vals.update(i)

x_tunables  = sp.Matrix([x_points[i+j] for i in range(1,len(x_points),6) for j in range(5)])
lbda_tunables  = sp.Matrix([lbda_points[1:-1][i] for i in range(n-2)])
tunables = sp.Matrix([x_tunables,lbda_tunables])


x_poly_0 = sp.poly(sum([x_polys[4][j] * lbda **(5-j) for j in range(6)]))
lbda_poly_0 = sp.poly(sum([lbda_polys[4][j] * t **(5-j) for j in range(6)]))

x_t_poly_0 = x_poly_0.subs(lbda,lbda_poly_0).subs(all_vals)
x_t_poly_0_d = sp.poly(x_t_poly_0.diff(t)).all_coeffs()

args = []
for i in range(len(x_polys)):
    x_poly_tmp = sp.poly(sum([x_polys[i][j] * lbda **(5-j) for j in range(6)]))
    lbda_poly_tmp = sp.poly(sum([lbda_polys[i][j] * t **(5-j) for j in range(6)]))
    x_t_poly_tmp = x_poly_tmp.subs(lbda,lbda_poly_tmp).subs(all_vals)
    args.append((x_t_poly_tmp, t<t_parts[i+1]))
f = sp.Piecewise(*args)

# plt.close('all')
# fig, ax = plt.subplots(3, 2, sharex='col')
# plot_piecewise(f,t,np.linspace(0,1,200), ax[0,0])
# plot_piecewise(f.diff(t,1),t,np.linspace(0,1,200), ax[1,0])
# plot_piecewise(f.diff(t,2),t,np.linspace(0,1,200), ax[1,1])
# plot_piecewise(f.diff(t,3),t,np.linspace(0,1,200), ax[2,0])
# plot_piecewise(f.diff(t,4),t,np.linspace(0,1,200), ax[2,1])

