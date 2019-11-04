#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:01:10 2019

@author: emil
"""


import numpy as np
from polynomial_utils import polydiff_d, get_bases, polyval, get_polys_from_bases
import numba as nb

@nb.njit(nb.types.Tuple((nb.float64[:,::1], nb.float64[::1]))(nb.float64[:], nb.float64[:],nb.float64[:], nb.float64[:,:]),
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



@nb.njit(nb.types.Tuple((nb.float64[:,::1], nb.float64[::1], nb.float64[:,::1]))(nb.float64[:], nb.float64[:], nb.float64[:], nb.int64),
         cache = True, fastmath = True)
def get_initial_path_1d(start, end, gates, tunables_n):
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



@nb.njit(nb.types.Tuple((nb.float64[:,:,::1], nb.float64[:],nb.float64[:,:,::1]))(nb.float64[:,::1],nb.float64[:,::1],nb.float64[:,::1]),
                        cache = True, parallel = True)
def get_initial_path(start, end, gates):
    tunables_n = 5
    s = np.empty((3, 8 + gates.shape[1] * 6, 6))
    tunables = np.empty((3, gates.shape[1] + 1 , tunables_n))
    lbda_points = np.empty((3, 19 + gates.shape[1] * 6))
    for i in nb.prange(3):
        s[i], lbda_points[i], tunables[i] = get_initial_path_1d(start[i], end[i], gates[i], tunables_n)
    return (s, lbda_points[0], tunables)

if __name__ == '__main__':
    a = np.array([get_initial_path_1d], dtype = type(get_initial_path_1d))
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    gates_n = 5
    start = np.zeros((3,4)).astype(np.float64)
    end = np.zeros((3,4)).astype(np.float64)
    gates = np.random.random((3,gates_n))

    s, lbda_points, tunables = get_initial_path(start, end, gates)
    p = 5

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(s.shape[1]):
        t0 = lbda_points[i + p]
        t1 = lbda_points[i + p + 1]
        print(t0)
        t_ = np.linspace(t0, t1, 100)
        ax.plot(polyval(s[0,i], t_),polyval(s[1,i], t_),polyval(s[2,i], t_))
        if (i-1)%6 == 0:
            ax.scatter(polyval(s[0,i], np.array([t0])),polyval(s[1,i], np.array([t0])),polyval(s[2,i], np.array([t0])))


