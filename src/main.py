#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:49:41 2019

@author: emil
"""


from accelerated.path import get_initial_path
import numpy as np



# if __name__ == '__main__':
#     gates_n = 5
#     start = np.zeros((3,4), dtype = np.float64)
#     end = np.zeros((3,4), dtype = np.float64)
#     gates = np.random.random((3,gates_n))
#     path, time_points, tunables = get_initial_path(start, end, gates)