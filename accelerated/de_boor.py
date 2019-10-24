#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:30:22 2019

@author: emil
"""


import numpy as np
from matplotlib import pyplot as plt

p = 5
n = 3
knots = np.arange(n)
ones = np.ones(p).astype(np.float64)
knots_pad = np.concatenate((ones*knots[0], knots, ones*knots[1]))
control_points = np.random.random(n)

plt.scatter(knots,control_points)