
import numpy as np
from numba import jitclass, float64

keys = [
('g', float64),
('delta', float64),
('m', float64),
('ix', float64),
('iy', float64),
('iz', float64),
('P_r', float64),
('D_r', float64),
('X', float64[:,:]),
]


@jitclass(keys)
class model(object):
    def __init__(self):
        self.g = 9.81
        self.delta = 0.001
        self.m = 1
        self.ix = 10
        self.iy = 0.25
        self.iz = 0.25
        self.P_r = 1
        self.D_r = 0
        self.X = np.array([[0], [0], [0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]).astype(np.float64)

    def iterate(self):
        ft, mx, my, mz = 0, 0, 0, 0
        p_t, q_t, r_t = 0, 0, 0
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z = self.X[:,0]
        x0 = np.tan(pitch)
        x1 = np.sin(roll)
        x2 = q*x1
        x3 = np.cos(roll)
        x4 = r*x3
        x5 = np.cos(pitch)
        x6 = 39.0*p
        x7 = np.sin(pitch)
        x8 = 9.81*x5
        x9 = np.cos(yaw)
        x10 = u*x5
        x11 = np.sin(yaw)
        x12 = x1*x11
        x13 = x3*x9
        x14 = x11*x3
        x15 = x1*x9
        X_d = np.array([[p + x0*x2 + x0*x4], [q*x3 - r*x1], [(x2 + x4)/x5], [(-p + p_t)/10], [-4.0*q + 4.0*q_t - r*x6], [q*x6 - 4.0*r + 4.0*r_t], [-q*w + r*v - 9.81*x7], [p*w - r*u + x1*x8], [-ft - p*v + q*u + x3*x8], [v*(-x14 + x15*x7) + w*(x12 + x13*x7) + x10*x9], [v*(x12*x7 + x13) + w*(x14*x7 - x15) + x10*x11], [-u*x7 + v*x1*x5 + w*x3*x5]])
        self.X += X_d * self.delta
        return self.X

roll = []
a = model()
for i in range(int(2e3)):
    roll.append(a.iterate()[3,0])
roll = np.array(roll)
