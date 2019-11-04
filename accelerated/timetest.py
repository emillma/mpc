#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:59:55 2019

@author: emil
"""


import time
from path import get_polys_from_tuning
import numba as nb
import numpy as np

@nb.njit(cache = True)
def timeit():
    for i in range(1000):
        start    = np.random.random(4)
        end      = np.random.random(4)
        gates_n  = 40
        tunables = np.random.random((gates_n+1,5)).astype(np.float64)
        gates    = np.random.random(gates_n)
        s, lbda_points = get_polys_from_tuning(start, end,gates, tunables)


if __name__ == '__main__':
    timeit()

    t = time.time()
    timeit()
    t2 = time.time()
    print(t2- t)