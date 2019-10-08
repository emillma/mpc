#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:05:36 2019

@author: emil
"""


from optimization_problem import get_interpolation_expressions
import sympy as sp
import numpy as np





if __name__ == '__main__':
    n = 5
    lbda = sp.symbols('lbda', real = True)
    x = list(np.linspace(0, 1, n))
    y = list(sp.symbols(['x_variabble_' + str(i) for i in range(n)]))
    start_cond = [0, 0, 0, 0]
    end_cond = [0, 0, 0, 0]
    polys, x, y = get_interpolation_expressions(x,y,start_cond,end_cond,lbda)

    args = []
    for i in range(len(polys)):
        args.append((polys[i], lbda<x[i+1]))
    f = sp.Piecewise(*args)