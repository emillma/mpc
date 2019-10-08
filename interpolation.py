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



def get_interpolation(x, y, start_cond, end_cond, lbda):
    n_points = len(x)+2
    solutions = sp.MatrixSymbol('solutions', n_points - 1, 6) #use as_mutable to get list
    for element in solutions:
        element._assumptions.update({"real":True, "commutative":True})
    x,y = x.copy(), y.copy()

    x.insert(1, (x[0] + x[1]) / 2.)
    x.insert(-1, (x[-1] + x[-2])/2.)

    y_is, y_ie = sp.symbols(['y_is', 'y_ie'], real = True, constant=True)
    # y[1] = y_is
    # y[-2] = y_ie
    y.insert(1, y_is)
    y.insert(-1, y_ie)
    potentials = sp.Matrix([1, lbda, lbda**2, lbda**3, lbda**4, lbda**5])
    polynomes = solutions * potentials
    problems = []

    problems += [polynomes[0,0].subs(lbda,x[0]) - y[0],
                 polynomes[0,0].subs(lbda,x[1]) - y[1]]

    problems += [polynomes[0,0].diff(lbda,j+1).subs(lbda,x[0])
                -start_cond[j] for j in range(3)]

    for i in range(1, n_points - 1):
        problems += [polynomes[i,0].subs(lbda,x[i]) - y[i],
                     polynomes[i,0].subs(lbda,x[i+1]) - y[i+1]]

        problems += [polynomes[i,0].diff(lbda,degree).subs(lbda,x[i])
                    -polynomes[i-1,0].diff(lbda,degree).subs(lbda,x[i])
                    for degree in range(1,5)]

    problems += [polynomes[n_points-2,0].diff(lbda,j+1).subs(lbda,x[n_points-1])
                -end_cond[j] for j in range(3)]

    variables  = [y_is, y_ie]
    variables += [solutions[i,j] for i in range(n_points-1) for j in range(6)]

    solution = list(sp.linsolve(problems, variables))[0]
    y1, y2 =  solution[:2]
    y[1] =  y1
    y[-2] = y2

    solution = solution[2:]
    out_polynomes = []
    for i in range(0, len(solution), 6):
        out_polynomes.append((sp.Matrix(solution[i:i + 6]).T * potentials)[0])
    return out_polynomes, x, y


lbda = sp.symbols('lbda', real = True)

ports = 1

x = list(np.linspace(0, 1, 3))
y = list(np.arange(3) + np.random.random(3) * 3)
y2 = list(sp.MatrixSymbol('solutions', 3, 1)) #use as_mutable to get list)
for element in y2:
    element._assumptions.update({"real":True, "commutative":True})

start_cond = [0, 0, 0, 0]
end_cond = [0, 0, 0, 0]

polys, x, y =get_interpolation(x,y,start_cond,end_cond, lbda)


args = []
for i in range(len(polys)):
    args.append((polys[i], lbda<x[i+1]))
f = sp.Piecewise(*args)
plt.close('all')
fig, ax = plt.subplots(3, 2, sharex='col')
plot_piecewise(f,lbda,np.linspace(0,1,200), ax[0,0])
plot_piecewise(f.diff(lbda,1),lbda,np.linspace(0,1,200), ax[1,0])
plot_piecewise(f.diff(lbda,2),lbda,np.linspace(0,1,200), ax[1,1])
plot_piecewise(f.diff(lbda,3),lbda,np.linspace(0,1,200), ax[2,0])
plot_piecewise(f.diff(lbda,4),lbda,np.linspace(0,1,200), ax[2,1])

ax[0,0].scatter(x,y)

