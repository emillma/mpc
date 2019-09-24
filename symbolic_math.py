#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:46:20 2019

@author: emil
"""
import sympy as sp
import re
from numbafy import numbafy

file = open('partials.txt', 'w')

g = sp.Symbol('g')
roll, pitch, yaw = sp.symbols(['roll', 'pitch', 'yaw'])
p, q, r = sp.symbols(['p', 'q', 'r'])
u, v, w = sp.symbols(['u', 'v', 'w'])
x, y, z = sp.symbols(['x', 'y', 'z'])
m, ix, iy, iz = sp.symbols(['m', 'ix', 'iy', 'iz'])
ft, tx, ty, tz = sp.symbols(['ft', 'tx', 'ty', 'tz'])


cos = sp.cos
sin = sp.sin
tan = sp.tan

roll_d     = p + r * cos(roll) * tan(pitch) + q * sin(roll) * tan(pitch)
pitch_d    = q * cos(roll) - r * sin(pitch)
yaw_d      = r * (cos(roll) / cos(pitch)) + q * (sin(roll) / cos(pitch))
p_d        = ((iy - iz) * r * q + tx) / ix
q_d        = ((iz - ix) * p * r + ty) / iy
r_d        = ((ix - iy) * p * q + tz) / iz
u_d        = r * v - q * w - g * sin(pitch)
v_d        = p * w - r * u + g * sin(roll) * cos(pitch)
w_d        = q * u - p * v + g * cos(pitch) * cos(roll) - ft / m

x_d        = (w * (sin(roll) * sin(yaw) + cos(roll) * cos(yaw) * sin(pitch))
                  -v * (cos(roll) * sin(yaw) - cos(yaw) * sin(roll) * sin(pitch))
                  +u * (cos(yaw) * cos(pitch)))

y_d        = (v * (cos(roll) * cos(yaw) + sin(roll) * sin(yaw) * sin(pitch))
                  -w * (cos(yaw) * sin(roll) - cos(roll) * sin(yaw) * sin(pitch))
                  +u * (cos(pitch) * sin(yaw)))

z_d        = (w * (cos(roll) * cos(pitch))
                  -u * (sin(pitch))
                  +v * (cos(pitch) * sin(roll)))

X = sp.Matrix([[roll],[pitch],[yaw],
                  [p],[q],[r],
                  [u],[v],[w],
                  [x],[y],[z]])

X_d = sp.Matrix([[roll_d],[pitch_d],[yaw_d],
                  [p_d],[q_d],[r_d],
                  [u_d],[v_d],[w_d],
                  [x_d],[y_d],[z_d]])














