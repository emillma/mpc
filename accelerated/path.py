#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:01:10 2019

@author: emil
"""


import numpy as np
from polynomial_utils import polydiff_d, get_bases, polyval, get_polys_from_bases
from matplotlib import pyplot as plt
import numba as nb

@nb.njit(nb.types.Tuple((nb.float64[:,::1], nb.float64[:]))(nb.float64[:], nb.float64[:],nb.float64[:], nb.float64[:,:]),
         cache = True, fastmath = True)
def get_path_from_tunables(start, end, gates, tunables):
# if True:
    tunables_n = tunables.shape[1]
    p = 5
    n = tunables.shape[0] * (tunables.shape[1]+1) - 1 + 4
    lbda_points = np.ones(n + 2*p).astype(np.float64)
    lbda_points = np.cumsum(lbda_points) - 1 - p

    bases = get_bases(lbda_points, p)

    scaling = np.zeros(bases.shape[0]).astype(np.float64)

    """
    Solve start problem
    """
    start_problem = np.zeros((4,4)).astype(np.float64)
    for i in range(4):
        for degree in range(4):
            start_problem[degree,i] = polyval(polydiff_d(bases[i,5-i], degree), lbda_points[p:p+1])[0]

    start_compensation = np.zeros(4).astype(np.float64)
    for i in range(4):
        start_compensation[i] = polyval(polydiff_d(bases[p-1,0], i), lbda_points[p:p+1])[0]

    A_start = np.ascontiguousarray(np.linalg.inv(start_problem))
    scaling[:4] = A_start@(start-start_compensation*tunables[0,0])
    """
    Solve end problem
    """
    problem_end = np.zeros((4,4)).astype(np.float64)
    for i in range(4):
        for degree in range(4):
            problem_end[degree,i] = polyval(polydiff_d(bases[i+bases.shape[0]-4,4-i], degree), lbda_points[-p-1:-p])[0]

    end_compensation = np.zeros(4).astype(np.float64)
    for i in range(4):
        end_compensation[i] = polyval(polydiff_d(bases[-p,-1], i), lbda_points[-p-1:-p])[0]


    A_end = np.ascontiguousarray(np.linalg.inv(problem_end))
    scaling[-4:] = A_end@(end-end_compensation*tunables[-1,-1])

    """
    Solve for gate points
    using the first bases around 0 for simplicity
    """
    for gate_point in nb.prange(tunables.shape[0]-1):
        gate_compensation = 0
        for i in range(4):
            gate_compensation += bases[i+i//2, 5-i-i//2,-1] *  tunables[gate_point + i // 2, (tunables_n-2+i)%tunables_n]

        scaling[3 + (tunables_n+1)*(gate_point+1)] = (gates[gate_point] - gate_compensation) / 0.55

    for i in nb.prange(tunables.shape[0]):
        scaling[4+(tunables_n+1)*i:4+(tunables_n+1)*(i+1)-1] =tunables[i]

    return get_polys_from_bases(bases, scaling, lbda_points, p), lbda_points

@nb.njit(nb.types.Tuple((nb.float64[:,::1], nb.float64[:], nb.float64[:,::1]))(nb.float64[:], nb.float64[:], nb.float64[:]),
         cache = True, fastmath = True)
def get_initial_path(start, end, gates):
    tunables_n = 5
    tunables = np.zeros((gates.shape[0] +1, tunables_n)).astype(np.float64)
    if gates.shape[0] == 0:
        tunables[0] = np.linspace(start[0], end[0], tunables_n +2)[1:-1]
    else:
        tunables[0] = np.linspace(start[0], gates[0], tunables_n +2)[1:-1]
        for i in range(1, gates.shape[0]):
            tunables[i] = np.linspace(gates[i-1], gates[i], tunables_n +2)[1:-1]
        tunables[-1] = np.linspace(gates[-1], end[0], tunables_n +2)[1:-1]

    s, lbda_points = get_path_from_tunables(start, end, gates, tunables)
    return s, lbda_points, tunables


if __name__ == '__main__':
    start    = np.array([0.,-1.,0.,0.], dtype = np.float64)
    end      = np.array([0.,0.,0.,0.], dtype = np.float64)
    gates_n = 2
    tunables = np.random.random((gates_n,5)).astype(np.float64)


    gates    = np.array([1.,2.,1.])

    s, lbda_points, tunables = get_initial_path(start, end, gates)
    p = 5

    plt.close('all')
    fig, ax = plt.subplots(5, 1, sharex = True)

    for t_i, poly in enumerate(s):
            t0 = lbda_points[t_i + p]
            t1 = lbda_points[t_i + p + 1]
            x_ = np.linspace(t0, t1, 100)
            ax[0].plot(x_,polyval(poly, x_))
            ax[1].plot(x_,polyval(polydiff_d(poly,1), x_))
            ax[2].plot(x_,polyval(polydiff_d(poly,2), x_))
            ax[3].plot(x_,polyval(polydiff_d(poly,3), x_))
            ax[4].plot(x_,polyval(polydiff_d(poly,4), x_))


