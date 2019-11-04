#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:51:31 2019

@author: emil
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_path(ax, path, time_points):
    p = 5
    for i in range(path.shape[1]):
        t0 = time_points[i + p]
        t1 = time_points[i + p + 1]
        print(t0)
        t_ = np.linspace(t0, t1, 100)
        ax.plot(np.polyval(path[0,i], t_),np.polyval(path[1,i], t_),np.polyval(path[2,i], t_))
        if (i-1)%6 == 0:
            ax.scatter(np.polyval(path[0,i], np.array([t0])),np.polyval(path[1,i], np.array([t0])),np.polyval(path[2,i], np.array([t0])))

