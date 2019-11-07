#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:22:23 2019

@author: emil
"""


import numpy as np
import numba as nb
from  accelerated.root_utils import get_polyroot_max
from accelerated.polynomial_utils import polyval, polydiff_2d, polypow2
from accelerated.path import get_initial_path

@nb.njit(nb.types.Tuple((nb.float64[::1], nb.float64[::1]))(nb.float64[:,:,::1],nb.float64[:]),
         fastmath = True, cache = True)
def get_max_values(path, lbda_points):
    p = 5
    out_arg = np.empty(path.shape[1])
    out_val = np.empty(path.shape[1])
    for i in range(path.shape[1]):
        polys = path[:,i,:]
        polys_diff = polydiff_2d(np.ascontiguousarray(polys))

        norm_squared = np.sum(polypow2(polys_diff), axis = 0)
        arg_max, val_max = get_polyroot_max(norm_squared,lbda_points[p+i:p+i+2])

        out_arg[i] = arg_max
        out_val[i] = val_max
    return out_arg, out_val




if __name__ == '__main__':
    from matplotlib import pyplot as plt
    gates_n = 5
    start = np.zeros((3,4)).astype(np.float64)
    end = np.zeros((3,4)).astype(np.float64)
    gates = np.random.random((3,gates_n))*10
    path, lbda_points, tunables = get_initial_path(start, end, gates)
    args, vals = get_max_values(path, lbda_points)
