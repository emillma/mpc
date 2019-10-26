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
from polynomial_utils import polyval, polydiff, B2polys


# def B(i, p, t, x_inter):
#     if p == 0:
#         # return 1
#         return sp.Piecewise((1, (x>=t[i]) & (x<t[i+1+p])), (0, True))
#     else:
#         return ((x-t[i])/(t[i+p] - t[i])) * B(i, p-1, t) + ((t[i+p+1] - x)/(t[i+p+1] - t[i+1])) * B(i+1, p-1, t)

def B(i, p, t, k):
    if p == 0:
        if k  >= i and k < i + 1:
            return 1
        else:
            return 0
    else:
        s = 0
        b0 = B(i, p-1, t,k)
        if b0 != 0:
            s+= b0 * ((x-t[i])/(t[i+p] - t[i]))
        b1 = B(i+1, p-1, t,k)
        if b1 != 0:
            s+= b1 * ((t[i+p+1] - x)/(t[i+p+1] - t[i+1]))
        return s


if __name__ == '__main__':
    plt.close('all')
    fig, ax = plt.subplots(3, 1, sharex = True)
    x = sp.symbols('x', real = True)
    n = 2

    for p in [2]:
        t_points = (np.arange(n + 2*p+1,dtype = np.float64) - p)
        # t_points[p + n//2:] += 5
        polys = np.zeros((n+p,p+1,p+1), dtype = np.float64)
        for i in range(0,n+p):
            for j, k in enumerate(range(i,min(i+p+1, n+p*2))):
                a = B(i,p,t_points, k)
                a_p = sp.Poly(a,x)
                polys[i, k-i, :] = a_p.all_coeffs()



    # for plot_i, data in enumerate(datas):
    # assert 0
    # s = B2polys(polys, np.ones(polys.shape[0]), t_points, 1)
    # c = np.ones(polys.shape[0])
    # ax[0].scatter(t_points[:-1], c)


    for t_i,p_tmp in enumerate(polys):
        for t_j in range(0,p+1):

                poly = p_tmp[t_j]
                t0 = t_points[t_i + t_j]
                t1 = t_points[t_i + t_j + 1]

                x_ = np.linspace(t0, t1, 100)
                y1 = polyval(poly, x_)
                poly_d = polydiff(poly)
                y2 = polyval(poly_d, x_)
                y3 = polyval(polydiff(poly_d), x_)
                ax[0].plot(x_, y1)
                ax[1].plot(x_, y2)
                ax[2].plot(x_, y3)
    # polys = data[0]








