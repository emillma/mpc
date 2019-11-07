#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:49:41 2019

@author: emil
"""


from accelerated.path import get_initial_path
from visualization.path_visualization import plot_path, plot_bases
from accelerated.spline_utils import get_bases
from accelerated.polynomial_utils import polydiff_2d

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    plt.close('all')
    lbda_points = np.ones((7), dtype = np.float64)
    lbda_points[2] = 10
    lbda_points = np.cumsum(lbda_points)-1
    bases = get_bases(lbda_points, 5)
    fig, ax = plt.subplots(2,1)
    plot_bases(ax[0], bases, lbda_points)
    bases_diff = polydiff_2d(bases[0])[None,:,:]
    # bases_diff = polydiff_2d(bases_diff[0])[None,:,:]
    plot_bases(ax[1], bases_diff, lbda_points)
