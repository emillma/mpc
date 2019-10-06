#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:48:04 2019

@author: emil
"""


import sympy as sp
import numpy as np
from utils import plot_piecewise

m = sp.MatrixSymbol('m', 3,1)
pa,pb,pc,pd,pe,pf = sp.symbols(['pa','pb','pc','pd','pe', 'pf'])
t = sp.symbols('t')
p5 = pa + pb*t + pc*t**2 + pd*t**3 + pe*t**4 + pf*t**5

p5_0 = p5.subs(t,0)
p5_1 = p5.subs(t,1)


problem = [p5.subs(t,0.),
           p5.diff(t,1).subs(t,0.),
           p5.diff(t,2).subs(t,0.),
           p5.diff(t,3).subs(t,0.),
           # p5.diff(t,4).subs(t,0.),
           p5.subs(t,1)-1]

s = sp.linsolve(problem, [pa,pb,pc,pd,pe,pf])


points = np.arange(5) + np.random.random(5)

# def get_interpolation(n=3):

n = 4
t = sp.symbols('t')
solutions = sp.MatrixSymbol('solutions', n -1, 6)
polynomes = solutions * sp.Matrix([1, t, t**2, t**3, t**4, t**5])

y = sp.MatrixSymbol('y', n,1)
x = np.linspace(0,1,n)
y = sp.Matrix(np.arange(n) + np.random.random(n))
start_cond = [10, 0, 0]
end_cond = [0, 0, 0]

problems = []
problems += [polynomes[0,0].subs(t,x[0]) - y[0],
             polynomes[0,0].subs(t,x[1]) - y[1],
             polynomes[0,0].diff(t,1).subs(t,x[0]) - start_cond[0],
             polynomes[0,0].diff(t,2).subs(t,x[0]) - start_cond[1]]
             # polynomes[0,0].diff(t,3).subs(t,x[0]) - start_cond[2]]
             # polynomes[0,0].diff(t,4).subs(t,x[0]),


for i in range(1, n - 1):
    problems += [polynomes[i,0].subs(t,x[i]) - y[i],
                 polynomes[i,0].subs(t,x[i+1]) - y[i+1],
                 polynomes[i,0].diff(t,1).subs(t,x[i]) - polynomes[i-1,0].diff(t,1).subs(t,x[i]),
                 polynomes[i,0].diff(t,2).subs(t,x[i]) - polynomes[i-1,0].diff(t,2).subs(t,x[i]),
                 polynomes[i,0].diff(t,3).subs(t,x[i]) - polynomes[i-1,0].diff(t,3).subs(t,x[i]),
                 polynomes[i,0].diff(t,4).subs(t,x[i]) - polynomes[i-1,0].diff(t,4).subs(t,x[i]),
                 # polynomes[i,0].diff(t,5).subs(t,x[i]) - polynomes[i-1,0].diff(t,5).subs(t,x[i-1])
                 ]

problems += [polynomes[n-2,0].subs(t,x[n-1]) - y[n-1],
             polynomes[n-2,0].diff(t,1).subs(t,x[n-1]) - end_cond[0],
             polynomes[n-2,0].diff(t,2).subs(t,x[n-1]) - end_cond[1]]
#              polynomes[n-2,0].diff(t,3).subs(t,x[n-1]) - end_cond[2]]

s = sp.linsolve(problems, [solutions[i,j] for i in range(n-1) for j in range(6) ])
s = list(s)[0]
tmp = sp.Matrix([1, t, t**2, t**3, t**4, t**5])
args = []
for i in range(n-1):
    p = (sp.Matrix(s[i*6:(i+1)*6]).T*tmp)[0,0]
    args.append((p, t<x[i+1]))

f = sp.Piecewise(*args)

plot_piecewise(f,t,np.linspace(0,1,1000))
# plot_piecewise(f.diff(t),t,np.linspace(0,1,1000))
# plot_piecewise(f.diff(t,2),t,np.linspace(0,1,1000))
# plot_piecewise(f.diff(t,3),t,np.linspace(0,1,1000))
# plot_piecewise(f.diff(t,3),t,np.linspace(0,1,1000))

plt.scatter(x,y)
