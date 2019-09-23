#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:46:20 2019

@author: emil
"""
import sympy
import re
file = open('partials.txt', 'w')


statespace = """
roll_d     = p + r * (cos_roll * tan_pitch) + q * (sin_roll * tan_pitch)
pitch_d    = q * (cos_roll) - r * (sin_pitch)
yaw_d      = r * (cos_roll / cos_pitch) + q * (sin_roll / cos_pitch)
p_d        = ((iy - iz) * r * q + tx) / ix
q_d        = ((iz - ix) * p * r + ty) / iy
r_d        = ((ix - iy) * p * q + tz) / iz
u_d        = r * v - q * w - g * sin_pitch
v_d        = p * w - r * u + g * (sin_roll * cos_pitch)
w_d        = q * u - p * v + g * (cos_pitch * cos_roll) - ft / m

x_d        = (w * (sin_roll * sin_yaw + cos_roll * cos_yaw * sin_pitch)
                  -v * (cos_roll * sin_yaw - cos_yaw * sin_roll * sin_pitch)
                  +u * (cos_yaw * cos_pitch))

y_d        = (v * (cos_roll * cos_yaw + sin_roll * sin_yaw * sin_pitch)
                  -w * (cos_yaw * sin_roll - cos_roll * sin_yaw * sin_pitch)
                  +u * (cos_pitch * sin_yaw))

z_d        = (w * (cos_roll * cos_pitch)
                  -u * (sin_pitch)
                  +v * (cos_pitch * sin_roll))
"""
statespace = re.sub('\n+', '\n', statespace) # removes smpty lines
statespace = re.sub('\n +', ' ', statespace) # removes lines not declearing variables
statespace = re.sub(' +', ' ', statespace)   # removes repeating spaces

find_replace = {'sin_roll':'sin(roll)',
                'cos_roll':'cos(roll)',
                'sin_pitch':'sin(pitch)',
                'cos_pitch':'cos(pitch)',
                'tan_pitch':'tan(pitch)',
                'sin_yaw':'sin(yaw)',
                'cos_yaw':'cos(yaw)'}

for k, v in find_replace.items():
    statespace = statespace.replace(k, v)

lines = re.split('\n', statespace)

lines = [line for line in lines if line]

variables = [line.split(' = ')[0] for line in lines]
variables = [v[:-2] for v in variables] #removes the _d
actuators = ['ft', 'tx', 'ty', 'tz']

expressions = [line.split(' = ')[1] for line in lines]


Jacobian = []
for line, name in zip(expressions, variables):
    tmp = []
    for variable in variables:
        tmp.append(sympy.simplify(sympy.diff(line, variable)))
    Jacobian.append(tmp)

Hessian = []
for line, name in zip(expressions, variables):
    tmp = []
    for variable in variables:
        tmp.append(sympy.simplify(sympy.diff(line, variable, 2)))
    Hessian.append(tmp)

B = []
for line, name in zip(expressions, variables):
    tmp = []
    for actuator in actuators:
        tmp.append(sympy.simplify(sympy.diff(line, actuator)))
    B.append(tmp)

System = []
for line, name in zip(expressions, variables):
    tmp = []
    tmp.append(sympy.simplify(line))
    System.append(tmp)

file.write('Variables \n\n')
for name in variables:
    file.write(name +', ')
file.write('\n\n\n')

file.write('System \n\n')
file.write('np.array([\n')
for i, liste in enumerate(System):
    file.write('[')
    for j, partial in enumerate(liste):
        for k, v in find_replace.items():
            partial = str(partial).replace(v, k)
        file.write(partial)
        file.write(', ')
    file.write('],\n')
file.write('], dtype = np.float64)\n\n\n')


file.write('Jacobian (A)\n\n')
file.write('np.array([\n')
for i, liste in enumerate(Jacobian):
    file.write('[')
    for j, partial in enumerate(liste):
        for k, v in find_replace.items():
            partial = str(partial).replace(v, k)
        file.write(partial)
        file.write(', ')
    file.write('],\n')
file.write('], dtype = np.float64)\n\n\n')


file.write('B\n\n')
file.write('np.array([\n')
for i, liste in enumerate(B):
    file.write('[')
    for j, partial in enumerate(liste):
        for k, v in find_replace.items():
            partial = str(partial).replace(v, k)
        file.write(partial)
        file.write(', ')
    file.write('],\n')
file.write('], dtype = np.float64)\n\n\n')


file.write('Hessian\n\n')
file.write('np.array([\n')
for i, liste in enumerate(Hessian):
    file.write('[')
    for j, partial in enumerate(liste):
        for k, v in find_replace.items():
            partial = str(partial).replace(v, k)
        file.write(partial)
        file.write(', ')
    file.write('],\n')
file.write('], dtype = np.float64)\n\n\n')
file.close()














