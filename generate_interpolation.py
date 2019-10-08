#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:48:45 2019

@author: emil
"""
import numpy as np
import sympy as sp
from optimization_problem import get_interpolation_expressions
from utils import sp_repr_1d, sp_repr_2d

def generate_model(expression,variable_names):

    file = open('./generated/interpolation.py', 'w')

    inputs = []
    for i, name in enumerate(variable_names):
        inputs.append(f'{name} = points[{i}]')
    code_constants = '\n    '.join(inputs)

    template = f"""
import numpy as np
from numba import njit, float64, prange


@njit(float64[::1](float64[:], float64[:], float64[::1]), parallel = True, fastmath=True)
def interpolate(points, x, out):
    {code_constants}
    for i in prange(x.shape[0]):
        lbda = x[i]
        out[i] = {sp.pycode(expression)}
    return out
"""
    file.write(template)
    file.close()
    return template
if __name__ == '__main__':
    n = 20
    lbda = sp.symbols('lbda', real = True)
    x = list(np.linspace(0, 1, n))
    y = list(sp.symbols(['x_variabble_' + str(i) for i in range(n)]))
    start_cond = [0, 0, 0, 0]
    end_cond = [0, 0, 0, 0]
    if 1:
        polys_out, x_out, y_out = get_interpolation_expressions(x,y,start_cond,end_cond,lbda)

    args = []
    for i in range(len(polys_out)):
        args.append((polys_out[i], lbda<x_out[i+1]))
    args.append((0, True))
    f = sp.Piecewise(*args)

    template = generate_model(f, y)
