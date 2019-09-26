#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:49:19 2019

@author: emil
"""


import numpy as np

def matrix_from_rpy(rpy):
    roll, pitch, yaw = rpy
    roll_matrix = np.array([[1, 0, 0],
                                [0, np.cos(roll), -np.sin(roll)],
                                [0, np.sin(roll), np.cos(roll)]], dtype = np.float64)

    pitch_matrix = np.array([[np.cos(roll), 0, np.sin(roll)],
                            [0, 1, 0],
                            [-np.sin(roll), 0, np.cos(roll)]], dtype = np.float64)

    yaw_matrix = np.array([[np.cos(roll), -np.sin(roll), 0],
                            [np.sin(roll), np.cos(roll), 0],
                            [0, 0, 1]], dtype = np.float64)

    R = yaw_matrix @ pitch_matrix @ roll_matrix
    return R

def subs_dict(a,b):
    dct = {}
    for i, item in enumerate(a):
        dct[item] = b[i]
    return dct