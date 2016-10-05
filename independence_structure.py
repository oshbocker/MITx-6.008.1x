# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:19:29 2016

@author: mysterion
"""

import numpy as np
import matplotlib.pyplot as plt
import comp_prob_inference

prob_W_I = np.array([[1/2, 0], [0, 1/6], [0, 1/3]])

prob_W = prob_W_I.sum(axis=1)
prob_I = prob_W_I.sum(axis=0)

print(np.outer(prob_W, prob_I))

prob_X_Y = np.array([[1/4, 1/4],[1/12, 1/12],[1/6,1/6]])
prob_X = prob_X_Y.sum(axis=1)
prob_Y = prob_X_Y.sum(axis=0)

print(np.outer(prob_W, prob_I))

# Ice Cream Exercise
prob_S_C = np.array([[0.4,0.1],[0.25,0.25]])
prob_S = prob_S_C.sum(axis=1)
prob_C = prob_S_C.sum(axis=0)

print(np.outer(prob_S,prob_C))

prob_S_C_0 = np.array([[0.72,0.08],[0.18,0.02]])
prob_S_C_1 = np.array([[0.08,0.12],[0.32,0.48]])

prob_S_0 = prob_S_C_0.sum(axis=1)
prob_C_0 = prob_S_C_0.sum(axis=0)

prob_S_1 = prob_S_C_1.sum(axis=1)
prob_C_1 = prob_S_C_1.sum(axis=0)

print(np.outer(prob_S_0,prob_C_0))
print(np.outer(prob_S_1,prob_C_1))