#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:04:42 2019

@author: emil
"""


import numpy as np
from accelerated import roots, splines

if __name__ == '__main__':
    n = 30
    x_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
    y_points = np.random.random(n)
    start_derivatives = np.array([0.,0.,0])
    end_derivatives = np.array([0.,0.,0])
    A,b, x_augmented = splines.get_spline_A_b(x_points, y_points,
                                   start_derivatives, end_derivatives)


    solution = np.linalg.solve(A,b)
    polys = solution[:-2].reshape(-1,6)
    for der in [1,2,3,4]:
        for i, poly in enumerate(polys):
            derivative = np.polynomial.polynomial.polyder(poly,der)
            print(roots.get_polyroots(derivative, x_augmented[i:i+2])) #use det and adj in sympy

