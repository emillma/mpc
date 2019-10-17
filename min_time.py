#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:04:42 2019

@author: emil
"""


import numpy as np
from spline_numpy import get_spline
from accelerated import roots
if __name__ == '__main__':
    n = 30
    x_points = np.arange(n)+((np.random.random(n)-0.5)*.4)
    y_points = np.random.random(n)
    start_derivatives = [0.,0.,0]
    end_derivatives = [0.,0.,0]
    polys, x_points_2 = get_spline(x_points, y_points,
                                   start_derivatives, end_derivatives)

    for i, poly in enumerate(polys):
        for der in [1,2,3,4]:
            derivative = np.polynomial.polynomial.polyder(poly,der)
            print(roots.get_polyroots(derivative, x_points_2[i:i+2])) #use det and adj in sympy

