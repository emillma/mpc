#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:36:46 2019

@author: emil
"""

import sympy as sp
import re
import numpy as np
from utils import sp_repr_1d, sp_repr_2d

def generate_model(expression, states, X0, gains, constants, K, use_cse=True):
    replacements = {'cos':'np.cos', 'sin':'np.sin', 'Matrix':'np.array', 'tan':'np.tan'}

    file = open('model.py', 'w')
    code_states = ''
    code_constants = ''
    code_cse = ''

    if states:
        code_states = ', '.join(f'{p}' for p in states)

    if constants:
        code_constants = []
        expression = expression.subs(constants)
        for k, v in constants.items():
            code_constants.append(f'self.{k} = {v}')

        code_constants = '\n        '.join(code_constants)



    if use_cse:
        expressions = sp.cse(expression, optimizations = 'basic')
        code_cse = []
        for e in expressions[0]:
            k, v = e
            code_cse.append(f'{k} = {v}')
        code_cse = '\n        '.join(code_cse)

        code_expression = expressions[1][0]
    else:
        code_expression = code_expression

    keys = []
    for k, v in constants.items():
        keys.append(f"('{k}', float64)")
    keys = ',\n'.join(keys)

    # X0 = np.array(X0, dtype = np.float64).reshape(-1)
    # K =  np.array(K, dtype = np.float64)
    # X_d = np.array(X_d, dtype = np.float64)

    template = f"""
import numpy as np
from numba import jitclass, float64

keys = [
{keys},
('X', float64[::1]),
('K', float64[:,::1]),
]


@jitclass(keys)
class model(object):
    def __init__(self):
        {code_constants}
        self.X = {sp_repr_1d(X0)}
        self.K = {sp_repr_2d(K)}

    def reset(self):
        self.X = {sp_repr_1d(X0)}

    def iterate(self):
        ft, mx, my, mz = 0, 0, 0, 0
        ft, p_t, q_t, r_t = -(self.K @ self.X)
        {code_states} = self.X
        {code_cse}
        X_d = {sp_repr_1d(code_expression)}
        self.X = self.X + X_d * self.delta
        return self.X

state = []
a = model()
for i in range(int(2e3)):
    state.append(a.iterate())
state = np.array(state)
"""

    for k, i in replacements.items():
        template = template.replace(k, i)
    file.write(template)
    file.close()
    return template

