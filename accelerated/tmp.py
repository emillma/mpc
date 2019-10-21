#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:20:20 2019

@author: emil
"""


import numpy as np
from max_derivatives import pow2
from matplotlib import pyplot as plt
from roots import get_polyroots
from polynomial_utils import  polydiff

def polyval(poly,x):
    # return x.reshape(-1,1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64))
    return np.sum(poly.reshape(-1,1)* x.reshape(1,-1)**(poly.shape[0]-1 - np.arange(poly.shape[0]).astype(np.float64)).reshape(-1,1), axis = 0)

i = -3
p = polys[i]
p = pow2(p)
bound = t_augmented[i-1:i+1]
x = np.linspace(bound[0],bound[1],1000)
plt.plot(x,polyval(p,x))
plt.plot(x,polyval(polydiff(p),x))

print(get_polyroots(polydiff(p), bound))
print(np.roots(polydiff(p)))

np.convolve