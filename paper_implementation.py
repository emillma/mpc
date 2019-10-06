#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:46:20 2019

@author: emil
"""
import sympy as sp
import re
from utils import subs_dict
# file = open('partials.txt', 'w')


"""
Constants
"""
g = sp.Symbol('g')
delta = sp.symbols('delta')
m, ix, iy, iz = sp.symbols(['m', 'ix', 'iy', 'iz'])
P_r, D_r = sp.symbols(['P_r', 'D_r'])


fx, fy, fz, mx, my, mz = sp.symbols(['fx', 'fy', 'fz', 'mx', 'my', 'mz'])
ft = sp.symbols('ft')

roll, pitch, yaw = sp.symbols(['roll', 'pitch', 'yaw'])
p, q, r = sp.symbols(['p', 'q', 'r'])
u, v, w = sp.symbols(['u', 'v', 'w'])
x, y, z = sp.symbols(['x', 'y', 'z'])

initials = {roll:0, pitch:0, yaw:0, p:0, q:0, r:0, u:0, v:0, w:0, x:0, y:0, z:0}

roll_d, pitch_d, yaw_d = sp.symbols(['roll_d', 'pitch_d', 'yaw_d'])
p_d, q_d, r_d = sp.symbols(['p_d', 'q_d', 'r_d'])
u_d, v_d, w_d = sp.symbols(['u_d', 'v_d', 'w_d'])
x_d, y_d, z_d = sp.symbols(['x_d', 'y_d', 'z_d'])


Rx, Ry, Rz, R = sp.symbols(['Rx', 'Ry', 'Rz', 'R'])

T = sp.symbols('T')

v, v_b, omega, omega_b = sp.symbols(['v', 'v_b', 'omega', 'omega_b'])

cos = sp.cos
sin = sp.sin
tan = sp.tan

constants = sp.Matrix([g, delta, m, ix, iy, iz])


#%% paper implementation
X = sp.Matrix([roll ,pitch, yaw,
                  p, q, r,
                  u, v, w,
                  x, y, z])

U = sp.Matrix([[ft, mx, my, mz]]).T

Rx = sp.Matrix([[1,             0,          0],
                [0,     cos(roll), -sin(roll)],
                [0,     sin(roll), cos(roll)]])

Ry = sp.Matrix([[cos(pitch),    0, sin(pitch)],
                [0,             1,          0],
                [-sin(pitch),   0, cos(pitch)]])

Rz = sp.Matrix([[cos(yaw), -sin(yaw),    0],
                [sin(yaw),  cos(yaw),    0],
                [0,                0,    1]])


R = Rz * Ry * Rx

T = sp.Matrix([[1, sin(roll)*tan(pitch),  cos(roll)*tan(pitch)],
               [0,            cos(roll),            -sin(roll)],
               [0, sin(roll)/cos(pitch), cos(roll)/cos(pitch)]])


v_b = sp.Matrix([u, v, w])
omega_b = sp.Matrix([p, q, r])

"""
v = [x_d, y_d, z_d]
v_b = [u, v, w]
omega = [ roll_d, pitch_d, yaw_d]
omega_b = [p, q, r]
"""
v       = R*v_b
omega   = T*omega_b

x_d, y_d, z_d = v
roll_d, pitch_d, yaw_d = omega


fb = R.T*sp.Matrix([0, 0, m*g]) - ft * sp.Matrix([0, 0, 1])
v_b_d = fb/m - omega_b.cross(v_b)
u_d, v_d, w_d = v_b_d

inertia = sp.diag(ix, iy, iz)
m_b = sp.Matrix([[mx, my, mz]]).T
omega_b_d = inertia.inv() * (m_b - omega_b.cross(inertia * omega_b))
p_d, q_d, r_d = omega_b_d

X_d = sp.Matrix([[roll_d],[pitch_d],[yaw_d],
                  [p_d],[q_d],[r_d],
                  [u_d],[v_d],[w_d],
                  [x_d],[y_d],[z_d]])

#%% custom stuff














