#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:24:30 2019

@author: emil
"""

tests = []
args = []
for i in range(len(problems)):
    for j in range(i+1,len(problems)):
        tests.append(sp.linsolve([problems[i], problems[j]], [solutions[i,j] for i in range(solutions.shape[0]) for j in range(6) ]))
        args.append((i,j))
for i in range(len(tests)):
    for j in range(i+1,len(tests)):
        if tests[i] == tests[j]:
            print(args[i],args[j])