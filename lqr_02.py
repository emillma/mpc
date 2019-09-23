#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:18:53 2019

@author: emil
"""

import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt
from statespace import quad
from visualizer import Visualizer

class System:
    def __init__(self, A, B, C, D, X, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.X = X
        self.delta = 1e-2
        self.state_number = A.shape[0]
        self.actuator_number = 4
        self.discretize(self.delta)
        self.lqr(Q, R)

    def get_pos_rpy(self):
        return self.X[-3:,0], self.X[:3,0]

    def discretize(self, delta = None):
        if delta:
            self.delta = delta

        self.A_d = np.eye(self.state_number) + self.A * delta
        self.B_d = (0.5 * delta**2 * self.A + self.delta * np.eye(self.state_number))@self.B
        self.C_d = self.C
        self.D_d = self.D

    def iterate(self, U = None):
        if U is None:
            U = np.zeros((self.actuator_number, 1))
        # print(U)
        self.X = self.A_d@self.X + self.B_d@U
        y = self.C_d@self.X + self.D_d@U
        # print(self.X)
        return self.get_pos_rpy()

    def set_x(self, X):
        self.X = X

    def get_optimal_gain(self):
        return -self.K@ self.X

    def simluate(self, time):
        out = np.empty((0,self.state_number))
        u = np.array([[0]])
        for i in np.arange(0,time,self.delta):
            u = -self.K@ self.X
            self.y = self.C_d@self.X + self.D_d@u
            out = np.vstack((out, np.squeeze(self.y)))
            self.iterate(u)
        return out


    def lqr(self, Q, R):
        """Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """
        #ref Bertsekas, p.151

        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(self.A, self.B, Q, R))
        # X = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, Q, R))

        #compute the LQR gain
        self.K = np.matrix(scipy.linalg.inv(R)@(self.B.T@X))

        # eigVals, eigVecs = scipy.linalg.eig(A-B@K)

if __name__ == '__main__':
    drone = quad(1,.5,.5,.5)
    A = drone.get_jacobian()
    B = drone.get_B()
    C = np.eye(12)
    D = np.zeros((12,4), dtype = np.float64)
    Q = np.diag([100,100,100,1,1,1,1,1,1,100,100,100])
    R = np.diag([1,1,1,1])*0.1

    X = np.zeros(12)[:,None]
    X[-3] = 1
    X[-1] = 1
    sys = System(A, B, C, D, X, Q, R)
    presentation = Visualizer(sys)
    presentation.animate(10)








