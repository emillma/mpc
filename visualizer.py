#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:54:02 2019

@author: emil
"""
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from utils import matrix_from_rpy
from matplotlib import animation


plt.close('all')

class Visualizer(object):
    def __init__(self, system):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Simulation')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 3)
        self.system = system
        self.line1, =  self.ax.plot([], [], [], c = 'b', lw=2)
        self.line2, =  self.ax.plot([], [], [], c = 'r', lw=2)

    def update_drawing(self, N):
        U = self.system.get_optimal_gain()
        pos, rpy = self.system.iterate(U = U)
        # print(pos,rpy)
        self.draw_drone(pos, rpy)
        return [self.line1, self.line2]

    def draw_drone(self, position, rpy):

        R = matrix_from_rpy(rpy)
        size = 0.2
        fl = R @ np.array([[ 1, 0,0]]).T * size + position
        fr = R @ np.array([[0,-1,0]]).T * size + position
        bl = R @ np.array([[0, 1,0]]).T * size + position
        br = R @ np.array([[-1,0,0]]).T * size + position
        # self.ax.scatter(position[0],position[1],position[2])
        self.line1.set_data([fl[0,0],br[0,0]],[fl[1,0],br[1,0]])
        self.line1.set_3d_properties([fl[2,0],br[2,0]])
        self.line2.set_data([fr[0,0],bl[0,0]],[fr[1,0],bl[1,0]])
        self.line2.set_3d_properties([fr[2,0],bl[2,0]])
        # self.ax.scatter(position[0],position[1],position[2])

    def animate(self, time):
        print(round(time/self.system.delta))
        self.ani = animation.FuncAnimation(self.fig, self.update_drawing,
                                      frames = round(time/self.system.delta),
                                      interval = round(self.system.delta*1000),
                                      repeat_delay = 500, blit=True)
        plt.show()

    def clear(self):
        # self.ax.clear()
        # pass
        self.update_drawing()

if __name__ == '__main__':
    pos = np.array([[0,1,2]]).T
    rpy = np.array([[0.1,0.1,1]]).T
    
    sim = Visualizer(1)
    sim.draw_drone(pos, rpy)