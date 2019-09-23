#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 14:39:30 2019

@author: emil
"""


import numpy as np

class quad(object):
    def __init__(self, m, ix, iy, iz, g = 9.81):
        """
        X = roll, pitch, yaw, p, q, r, u, v, w, x, y, z,
        """
        self.m = float(m)
        self.ix = float(ix)
        self.iy = float(iy)
        self. iz = float(iz)
        self.g = float(g)
        self.I = np.diag([ix, iy, iz])

        self.X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype = np.float64).T


    def prepare_variables(self, X):
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
        return (ix, iy, iz, g, m,
                roll, pitch, yaw, p, q, r, u, v, w, x, y, z,
                sin_roll, cos_roll, sin_pitch, cos_pitch,
                tan_pitch, sin_yaw, cos_yaw)

    def get_X_dot(self, U = np.array([0,0,0,0], dtype = np.float64), X = None):
        (ix, iy, iz, g, m,
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z,
        sin_roll, cos_roll, sin_pitch, cos_pitch,
        tan_pitch, sin_yaw, cos_yaw) = self.prepare_variables(X)

        ft, tx, ty, tz = U
        X_dot = np.array([
        [p + q*sin_roll*tan_pitch + r*cos_roll*tan_pitch, ],
        [q*cos_roll - r*sin_pitch, ],
        [(q*sin_roll + r*cos_roll)/cos_pitch, ],
        [(q*r*(iy - iz) + tx)/ix, ],
        [(-p*r*(ix - iz) + ty)/iy, ],
        [(p*q*(ix - iy) + tz)/iz, ],
        [-g*sin_pitch - q*w + r*v, ],
        [g*sin_roll*cos_pitch + p*w - r*u, ],
        [-ft/m + g*cos_pitch*cos_roll - p*v + q*u, ],
        [u*cos_pitch*cos_yaw + v*(sin_pitch*sin_roll*cos_yaw - sin_yaw*cos_roll) + w*(sin_pitch*cos_roll*cos_yaw + sin_roll*sin_yaw), ],
        [u*sin_yaw*cos_pitch + v*(sin_pitch*sin_roll*sin_yaw + cos_roll*cos_yaw) + w*(sin_pitch*sin_yaw*cos_roll - sin_roll*cos_yaw), ],
        [-u*sin_pitch + v*sin_roll*cos_pitch + w*cos_pitch*cos_roll, ],
        ], dtype = np.float64)
        return X_dot

    def get_jacobian(self, X = None):
        (ix, iy, iz, g, m,
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z,
        sin_roll, cos_roll, sin_pitch, cos_pitch,
        tan_pitch, sin_yaw, cos_yaw) = self.prepare_variables(X)

        out = np.array([
        [(q*cos_roll - r*sin_roll)*tan_pitch, (q*sin_roll + r*cos_roll)/cos_pitch**2, 0, 1, sin_roll*tan_pitch, cos_roll*tan_pitch, 0, 0, 0, 0, 0, 0, ],
        [-q*sin_roll, -r*cos_pitch, 0, 0, cos_roll, -sin_pitch, 0, 0, 0, 0, 0, 0, ],
        [(q*cos_roll - r*sin_roll)/cos_pitch, (q*sin_roll + r*cos_roll)*sin_pitch/cos_pitch**2, 0, 0, sin_roll/cos_pitch, cos_roll/cos_pitch, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, r*(iy - iz)/ix, q*(iy - iz)/ix, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, -r*(ix - iz)/iy, 0, -p*(ix - iz)/iy, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, q*(ix - iy)/iz, p*(ix - iy)/iz, 0, 0, 0, 0, 0, 0, 0, ],
        [0, -g*cos_pitch, 0, 0, -w, v, 0, r, -q, 0, 0, 0, ],
        [g*cos_pitch*cos_roll, -g*sin_pitch*sin_roll, 0, w, 0, -u, -r, 0, p, 0, 0, 0, ],
        [-g*sin_roll*cos_pitch, -g*sin_pitch*cos_roll, 0, -v, u, 0, q, -p, 0, 0, 0, 0, ],
        [v*(sin_pitch*cos_roll*cos_yaw + sin_roll*sin_yaw) - w*(sin_pitch*sin_roll*cos_yaw - sin_yaw*cos_roll), (-u*sin_pitch + v*sin_roll*cos_pitch + w*cos_pitch*cos_roll)*cos_yaw, -u*sin_yaw*cos_pitch - v*(sin_pitch*sin_roll*sin_yaw + cos_roll*cos_yaw) - w*(sin_pitch*sin_yaw*cos_roll - sin_roll*cos_yaw), 0, 0, 0, cos_pitch*cos_yaw, sin_pitch*sin_roll*cos_yaw - sin_yaw*cos_roll, sin_pitch*cos_roll*cos_yaw + sin_roll*sin_yaw, 0, 0, 0, ],
        [v*(sin_pitch*sin_yaw*cos_roll - sin_roll*cos_yaw) - w*(sin_pitch*sin_roll*sin_yaw + cos_roll*cos_yaw), (-u*sin_pitch + v*sin_roll*cos_pitch + w*cos_pitch*cos_roll)*sin_yaw, u*cos_pitch*cos_yaw + v*(sin_pitch*sin_roll*cos_yaw - sin_yaw*cos_roll) + w*(sin_pitch*cos_roll*cos_yaw + sin_roll*sin_yaw), 0, 0, 0, sin_yaw*cos_pitch, sin_pitch*sin_roll*sin_yaw + cos_roll*cos_yaw, sin_pitch*sin_yaw*cos_roll - sin_roll*cos_yaw, 0, 0, 0, ],
        [(v*cos_roll - w*sin_roll)*cos_pitch, -u*cos_pitch - v*sin_pitch*sin_roll - w*sin_pitch*cos_roll, 0, 0, 0, 0, -sin_pitch, sin_roll*cos_pitch, cos_pitch*cos_roll, 0, 0, 0, ],
        ], dtype = np.float64)
        return out

    def get_B(self, X = None):
        (ix, iy, iz, g, m,
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z,
        sin_roll, cos_roll, sin_pitch, cos_pitch,
        tan_pitch, sin_yaw, cos_yaw) = self.prepare_variables(X)

        out = np.array([
        [0, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        [0, 1/ix, 0, 0, ],
        [0, 0, 1/iy, 0, ],
        [0, 0, 0, 1/iz, ],
        [0, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        [-1/m, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        [0, 0, 0, 0, ],
        ], dtype = np.float64)
        return out







# drone = quad(1,.5,.5,.5)
# for i in range(10000):
#     drone.get_X_dot()
# A = drone.get_jacobian(X = np.random.random(12)*0)
