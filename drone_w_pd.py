#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:46:43 2019

@author: emil
"""
import sympy as sp
from paper_implementation import X, X_d, U, constants
from utils import subs_dict
import scipy
import numpy as np
from generate_model import generate_model

P_r, D_r = sp.symbols(['P_r', 'D_r'])
omega_b_t = sp.symbols(['omega_b_t'])
p_t, q_t, r_t = sp.symbols(['p_t', 'q_t', 'r_t'])


ft, mx, my, mz = U
roll, pitch, yaw, p, q, r, u, v, w, x, y, z = X
roll_d, pitch_d, yaw_d, p_d, q_d, r_d, u_d, v_d, w_d, x_d, y_d, z_d = X_d
g, delta, m, ix, iy, iz = constants


omega_b_d = sp.Matrix([[q_d, p_d, r_d]]).T
omega_b = sp.Matrix([[p, q, r]]).T


U2 = sp.Matrix([[ft, p_t, q_t, r_t]]).T
target_rates = sp.Matrix([[p_t, q_t, r_t]]).T

rotation = sp.Matrix([[roll, pitch, yaw, p, q, r]]).T

#%%Code

omega_b_dd = omega_b_d.jacobian(rotation)*rotation
m_internal = -P_r*(omega_b - target_rates) - D_r*omega_b_dd
mx_i, my_i, mz_i = m_internal
replacements = {mx:mx_i, my:my_i, mz:mz_i}
omega_b_d_external = omega_b_d.subs(replacements)
p_d2, q_d2, r_d2 = omega_b_d_external


X_d2 = sp.Matrix([[roll_d],[pitch_d],[yaw_d],
                  [p_d2],[q_d2],[r_d2],
                  [u_d],[v_d],[w_d],
                  [x_d],[y_d],[z_d]])

A2 = X_d2.jacobian(X)

#%%Find LQR controller
X0 = sp.Matrix([[0,0,0,0,0,0,0,0,0,0,0,0]]).T

constants = {g:9.81*1, delta:1e-3, m:1,ix:0.25, iy:0.25, iz:0.25, P_r:1, D_r:0}
X0_dict = subs_dict(X,X0)

A = X_d2.jacobian(X)
B = X_d2.jacobian(U2)

A = A.subs(constants).subs(X0_dict)
B = B.subs(constants).subs(X0_dict)

Q_lqr = sp.diag(10,10,1,1,1,1,.1,.1,.1,100,100,100)
R_lqr = sp.diag(0.1,.1,.1,.1)
N = sp.zeros(12,4)
# P = sp.solve(A.T*P+P*A-(P*B+N)*R.inv()*(B.T*P+N.T)+Q = 0)

A_array = np.array(A, dtype = np.float64)
B_array = np.array(B, dtype = np.float64)
Q_array = np.array(Q_lqr, dtype = np.float64)
R_array = np.array(R_lqr, dtype = np.float64)

P_lqr = scipy.linalg.solve_continuous_are(A_array,B_array,Q_array,R_array)
P_lqr = sp.Matrix(P_lqr)

K = R_lqr.inv()*(B.T*P_lqr+N.T)
K = sp.Matrix(np.where(np.less(K,1e-3), 0., K))
#%%Generate model

generate_model(X_d2, X0 = X0, states = X, gains = U2, constants = constants, K = K)








