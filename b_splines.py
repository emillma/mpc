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
    fig, ax = plt.subplots(3, 1, sharex = True)
    x = sp.symbols('x', real = True)
    p =5
    n = 6+p

    t_points = (np.ones(n + 2*p,dtype = np.float64))
    t_points[p+4] *=5

    t_points = np.cumsum(t_points) - 1 - p

    polys = get_bases(t_points, p)

    c = np.arange(polys.shape[0], dtype= np.float64)
    c[:4] = np.array([ 0.85586735, -0.13520408, -2.86600765, -2.71194728])
    s = get_polys_from_bases(polys, c, t_points, p)

    for t_i, poly in enumerate(s):
        t0 = t_points[t_i + p]
        t1 = t_points[t_i + p + 1]
        x_ = np.linspace(t0, t1, 100)
        print(t0)
        ax[0].plot(x_,polyval(polydiff(poly), x_))



    for t_i,p_tmp in enumerate(polys):
        for t_j in range(0,p+1):

                poly = p_tmp[t_j]
                t0 = t_points[t_i + t_j]
                t1 = t_points[t_i + t_j + 1]

                x_ = np.linspace(t0, t1, 100)
                y1 = polyval(poly, x_)

                ax[1].plot(x_, y1)
                for d in range(2):
                    p_d = polydiff_d(poly, d)
                    y = polyval(p_d, x_)
                    ax[1+d].plot(x_, y)







