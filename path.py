#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:11:08 2019

@author: emil
"""


import numpy as np
from matplotlib import pyplot as plt
if 0:
    from accelerated.polynomial_utils import get_bases, polyval, get_polys_from_bases, polydiff_d
n = 2
p = 5
lbda_points = np.arange(n+2*p, dtype = np.float64) - p
bases = get_bases(lbda_points, 5)

c = np.arange(bases.shape[0])
x = np.arange(8, dtype = np.float64)

first = bases[np.arange(4), 5-np.arange(4)]

problem = np.zeros((4,4), dtype = np. float64)
for i in range(4):
    for j in range(4):
        problem[i,j] = polyval(polydiff_d(first[i], j), np.array([0.]))