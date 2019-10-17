#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:31:27 2019

@author: emil
"""


import numpy as np
from numba import njit, float64, void

@njit(void(float64[::-1]))
def add(a):
    a[0] = 23.

b = np.empty((1,))
add(b)
