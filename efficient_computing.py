# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 06:59:48 2016

@author: mysterion
"""

import numpy as np

p_X = {}
for x in X_alphabet:
    total = 0
    for y in Y_alphabet:
        total = total + p_XY[x][y]
    p_X[x] = total