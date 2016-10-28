# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:04:35 2016

@author: mysterion
"""

import numpy as np

A = np.array([[0.25, 0.75, 0],[0,0.25,0.75],[0,0,1]])
B = np.array([[1,0],[0,1],[1,0]])

pi_0 = np.array([1/3,1/3,1/3])

forward_message_1 = (B[:,0]*pi_0) @ A
print(forward_message_1)