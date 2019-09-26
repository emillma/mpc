
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
('X', float64[::1]),
('K', float64[:,::1]),
]


@jitclass(keys)
class model(object):
    def __init__(self):
        self.g = 9.81
        self.delta = 0.001
        self.m = 1
        self.ix = 0.25
        self.iy = 0.25
        self.iz = 0.25
        self.P_r = 1
        self.D_r = 0
        self.X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.K = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 42.2195700298250, 0.0, 0.0, 5.66654965697051, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [42.2195700298249, 0.0, 0.0, 5.66654965697049, 0.0, 0.0, 0.0, 16.5285101433270, 0.0, 0.0, 31.6227766016838, 0.0], [0.0, 0.0, 3.16227766016838, 0.0, 0.0, 2.54699010854049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype = np.float64)

    def reset(self):
        self.X = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    def iterate(self):
        ft, mx, my, mz = 0, 0, 0, 0
        ft, p_t, q_t, r_t = -(self.K @ self.X)
        roll, pitch, yaw, p, q, r, u, v, w, x, y, z = self.X
        x0 = np.tan(pitch)
        x1 = np.sin(roll)
        x2 = q*x1
        x3 = np.cos(roll)
        x4 = r*x3
        x5 = np.cos(pitch)
        x6 = np.sin(pitch)
        x7 = 9.81*x5
        x8 = np.cos(yaw)
        x9 = u*x5
        x10 = np.sin(yaw)
        x11 = x1*x10
        x12 = x3*x8
        x13 = x10*x3
        x14 = x1*x8
        X_d = np.array([p + x0*x2 + x0*x4, q*x3 - r*x1, (x2 + x4)/x5, 4.0*(-q + q_t), 4.0*(-p + p_t), 4.0*(-r + r_t), -q*w + r*v - 9.81*x6, p*w - r*u + x1*x7, -ft - p*v + q*u + x3*x7, v*(-x13 + x14*x6) + w*(x11 + x12*x6) + x8*x9, v*(x11*x6 + x12) + w*(x13*x6 - x14) + x10*x9, -u*x6 + v*x1*x5 + w*x3*x5], dtype=np.float64)
        self.X = self.X + X_d * self.delta
        return self.X

state = []
a = model()
for i in range(int(2e7)):
    state.append(a.iterate())
state = np.array(state)
