#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:31:27 2019

@author: emil
"""


import numpy as np
from numba import njit, float64, void

@njit
def add(a):
    b = a
    b[:] = np.array([3])

a = np.zeros((1,))
add(a)
