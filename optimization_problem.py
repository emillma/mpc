#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:48:04 2019

@author: emil
"""


import sympy as sp
import numpy as np
from utils import plot_piecewise
from matplotlib import pyplot as plt





def get_interpolation_expressions(x, y, start_cond, end_cond, lbda):
    n_points = len(x)+2
    solutions = sp.MatrixSymbol('solutions', n_points - 1, 6) #use as_mutable to get list
    for element in solutions:
        element._assumptions.update({"real":True})
    x,y = x.copy(), y.copy()

    x.insert(1, (x[0] + x[1]) / 2.)
    x.insert(-1, (x[-1] + x[-2])/2.)

    y_is, y_ie = sp.symbols(['y_is', 'y_ie'], real = True)
    # y[1] = y_is
    # y[-2] = y_ie
    y.insert(1, y_is)
    y.insert(-1, y_ie)
    potentials = sp.Matrix([1, lbda, lbda**2, lbda**3, lbda**4, lbda**5])
    for element in potentials:
        element._assumptions.update({"real":True})
    polynomes = solutions * potentials
    problems = []
    problems += [polynomes[0,0].subs(lbda,x[0]) - y[0],
                  polynomes[0,0].subs(lbda,x[1]) - y[1]]

    problems += [polynomes[0,0].diff(lbda,j).subs(lbda,x[0])
                -start_cond[j] for j in range(1, 4)]

    for i in range(1, n_points - 1):
        problems += [polynomes[i,0].subs(lbda,x[i]) - y[i],
                     polynomes[i,0].subs(lbda,x[i+1]) - y[i+1]]

        problems += [polynomes[i,0].diff(lbda,degree).subs(lbda,x[i])
                    -polynomes[i-1,0].diff(lbda,degree).subs(lbda,x[i])
                    for degree in range(1,5)]

    problems += [polynomes[n_points-2,0].diff(lbda,j).subs(lbda,x[n_points-1])
                -end_cond[j] for j in range(1, 4)]

    variables  = [y_is, y_ie]
    variables += [solutions[i,j] for i in range(n_points-1) for j in range(6)]

    solution = list(sp.linsolve(problems, variables))[0]

    solution = solution[2:]
    out_polynomes = []
    for i in range(0, len(solution), 6):
        out_polynomes.append(sp.Tuple(*[solution[i+5-j] for j in range(6)]))

    return out_polynomes, x

def get_problem(checkpoints_n, tunables_n = 5):
    points_n = 2 + checkpoints_n + tunables_n * (checkpoints_n + 1)

    lbda_parts = list(np.linspace(0,1,points_n))
    x_start = sp.symbols([f'x_start_{i}' for i in range(4)], real = True)
    x_end = sp.symbols([f'x_end_{i}' for i in range(4)], real = True)
    x_points = [x_start[0]]
    for i in range(checkpoints_n+1):
        for j in range(tunables_n):
            x_points.append(sp.symbols(f'x_part_{i}_tunable_{j}', real = True))
        x_points.append(sp.symbols(f'x_checkpoint_{i}', real = True))
    x_points[-1] = x_end[0]
    lbda = sp.symbols('lbda')

    t_parts = list(np.linspace(0,1,points_n))
    lbda_start = [0.,1.,0.,0.]
    lbda_end = [1.,1.,0.,0.]
    lbda_points = [lbda_start[0]]
    for i in range(checkpoints_n+1):
        for j in range(tunables_n):
            lbda_points.append(sp.symbols(f'lbda_part_{i}_tunable_{j}', real = True))
        lbda_points.append(sp.symbols(f'lbda_checkpoint_{i}', real = True))
    lbda_points[-1] = lbda_end[0]
    t = sp.symbols('t')

    x_polys, lbda_parts = get_interpolation_expressions(lbda_parts, x_points, x_start, x_end, lbda)
    lbda_polys, t_parts = get_interpolation_expressions(t_parts, lbda_points, lbda_start, lbda_end, t)
    return (x_polys, lbda_parts, x_points, x_start, x_end, lbda), (lbda_polys, t_parts, lbda_points, t)













