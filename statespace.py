#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:39:30 2019

@author: emil
"""


import numpy as np
import scipy.signal as signal
import scipy
from matplotlib import pyplot as plt
import quaternion
from numba import jitclass
@jitclass
class quad(object):
    def __init__(self, m, ix, iy, iz, g = 9.81):
        """
        X = x, y, z, x', y', z', roll, pitch, yaw, roll', pitch', yaw'

        _w = 
        _f = frame
        _d = derivative
        _dd = double derivative
        _s = system
        """
        self.m = float(m)
        self.ix = float(ix)
        self.iy = float(iy)
        self. iz = float(iz)
        self.g = float(g)
        self.I = np.diag([ix, iy, iz])



        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.
        self.p = 0.
        self.q = 0.
        self.r = 0.
        self.u = 1.
        self.v = 0.
        self.w = 0.
        self.x = 0.
        self.y = 0.
        self.z = 0.

        self.X = np.array([[self.roll,
                             self.pitch,
                             self.yaw,
                             self.p,
                             self.q,
                             self.r,
                             self.u,
                             self.v,
                             self.w,
                             self.x,
                             self.y,
                             self.z]], dtype= np.float64).T

    def get_X_dot(self, ft=0, tx=0, ty=0, tz=0, X = None):
        ix      = self.ix
        iy      = self.iy
        iz      = self.iz
        g = self.g
        m = self.m

        if X is None:
            roll, pitch, yaw, p, q, r, u, v, w, x, y, z = self.X

        else:
            roll, pitch, yaw, p, q, r, u, v, w, x, y, z = X


        sin_roll    = np.sin(roll)
        cos_roll    = np.cos(roll)
        sin_pitch   = np.sin(pitch)
        cos_pitch   = np.cos(pitch)
        tan_pitch   = np.tan(pitch)
        sin_yaw     = np.sin(yaw)
        cos_yaw     = np.cos(yaw)

        self.roll_d     = p + r * (cos_roll * tan_pitch) + q * (sin_roll * tan_pitch)
        self.pitch_d    = q * (cos_roll) - r * (sin_pitch)
        self.yaw_d      = r * (cos_roll / cos_pitch) + q * (sin_roll / cos_pitch)
        self.p_d        = ((iy - iz) * r * q + tx) / ix
        self.q_d        = ((iz - ix) * p * r + ty) / iy
        self.r_d        = ((ix - iy) * p * q + tz) / iz
        self.u_d        = r * v - q * w - g * sin_pitch
        self.v_d        = p * w - r * u + g * (sin_roll * cos_pitch)
        self.w_d        = q * u - p * v + g * (cos_pitch * cos_roll) - ft / m

        self.x_d        = (w * (sin_roll * sin_yaw + cos_roll * cos_yaw * sin_pitch)
                          -v * (cos_roll * sin_yaw - cos_yaw * sin_roll * sin_pitch)
                          +u * (cos_yaw * cos_pitch))

        self.y_d        = (v * (cos_roll * cos_yaw + sin_roll * sin_yaw * sin_pitch)
                          -w * (cos_yaw * sin_roll - cos_roll * sin_yaw * sin_pitch)
                          +u * (cos_pitch * sin_yaw))

        self.z_d        = (w * (cos_roll * cos_pitch)
                          -u * (sin_pitch)
                          +v * (cos_pitch * sin_roll))

        self.X_d = np.array([[self.roll_d,
                             self.pitch_d,
                             self.yaw_d,
                             self.p_d,
                             self.q_d,
                             self.r_d,
                             self.u_d,
                             self.v_d,
                             self.w_d,
                             self.x_d,
                             self.y_d,
                             self.z_d]], dtype= np.float64).T

        return self.X_d.copy()

    # def get_linearized(self):






drone = quad(1,.5,.5,.5)
for i in range(1000):
    drone.get_X_dot()
print(drone.X_d)
