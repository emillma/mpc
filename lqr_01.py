#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:20:41 2019

@author: emil
"""


import numpy as np
import scipy.signal as signal
import scipy
from matplotlib import pyplot as plt

class system:
    def __init__(self):
        self.A = np.array([[0,1],[-1,0]])
        self.B = np.array([0,1])[None].T
        self.C = np.array([[1,0],[0,1]])
        self.D = np.array([[0,0]]).T
        self.x = np.array([0,0])[None].T
        self.delta = 1e-2
        self.discretize(self.delta)
        self.lqr(np.eye(2)*100,np.eye(1))
    def discretize(self, delta):
        self.delta = delta
        self.Ad = scipy.linalg.expm(self.A*delta)
        shape = self.A.shape
        self.Bd = np.linalg.inv(self.A)@(scipy.linalg.expm(self.A*delta)-np.eye(shape[0],shape[1]))@self.B
        # (scipy.linalg.expm(self.A*delta)-np.eye(shape[0],shape[1]))@self.B
        self.Cd = self.C
        self.Dd = self.D

    def iterate(self, u):

        self.x = self.Ad@self.x + self.Bd@u
        y = self.Cd@self.x + self.Dd@u

    def simluate(self, init, time):
        out = np.empty((0,2))
        self.x = init
        u = np.array([[0]])
        for i in np.arange(0,time,self.delta):
            u = -self.K@self.x
            self.y = self.Cd@self.x + self.Dd@u
            out = np.vstack((out, np.squeeze(self.y)))
            self.iterate(u)
        return out



    def lqr(self, Q,R):
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
         

sys = system()
out = sys.simluate(np.array([[1],[0]]), 10)
plt.plot(out[:,0])