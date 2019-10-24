#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:41:13 2019

@author: emil
"""

from scipy.interpolate import BSpline
from  matplotlib import pyplot  as plt
import numpy as np

k = 5
t = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
c = [1,1,1,1,1,1,1,1,1,1]
spl = BSpline(t, c, k)
spl.basis_element(1)