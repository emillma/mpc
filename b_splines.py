#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:41:13 2019

@author: emil
"""
import sys
sys.path.insert(1, 'accelerated')
import numpy as np
from matplotlib import pyplot as plt
import sympy as sp
from polynomial_utils import polyval, polydiff, get_polys_from_bases, get_bases, polydiff_d


if __name__ == '__main__':
    plt.close('all')
    fig, ax = plt.subplots(2, 1, sharex = True)
    x = sp.symbols('x', real = True)
    n = 5
    p =5

    t_points = (np.arange(n + 2*p,dtype = np.float64) - p)
    t_points[2*p:] += 5

    polys = get_bases(t_points, p)

    c = np.arange(polys.shape[0], dtype= np.float64)
    s = get_polys_from_bases(polys, c, t_points, p)

    for t_i, poly in enumerate(s):
        t0 = t_points[t_i + p]
        t1 = t_points[t_i + p + 1]
        x_ = np.linspace(t0, t1, 100)
        ax[0].plot(x_,polyval(polydiff(poly), x_))



    for t_i,p_tmp in enumerate(polys):
        for t_j in range(0,p+1):

                poly = p_tmp[t_j]
                t0 = t_points[t_i + t_j]
                t1 = t_points[t_i + t_j + 1]

                x_ = np.linspace(t0, t1, 100)
                y1 = polyval(poly, x_)

                ax[1].plot(x_, y1)
                for d in range(1):
                    p_d = polydiff_d(poly, d)
                    y = polyval(p_d, x_)
                    ax[1+d].plot(x_, y)







