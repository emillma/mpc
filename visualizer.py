#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:54:02 2019

@author: emil
"""
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
plt.close('all')
class visualizer(object):
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(0, 10)

    def draw2d(self, pose):
        x = pose[0]
        y = pose[1]
        z = pose[2]
        r = pose[3]
        arm=.5
        center = np.array([x,y,z])
        front_right = np.array([x + arm, y + arm*np.cos(r), z + arm*np.sin(r)])
        back_left = np.array([x - arm, y - arm*np.cos(r), z - arm*np.sin(r)])

        for p, c in zip([center, front_right, back_left],['b','g','r']):
            self.ax.scatter(p[0], p[1], p[2])


class quad:
    x = np.array([1,2,2,np.pi/6.])


sim = visualizer()
sim.draw2d(quad())