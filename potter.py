# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:58:24 2016

@author: mysterion
"""
import numpy as np


x4_x2 = np.matrix([[2/4,1/4],[0,0]])
x2_x1 = np.matrix([[2/4,1/4],[2/4,3/4]])

print(x4_x2*x2_x1)
(x4_x2*x2_x1).sum(axis=1).sum(axis=0)[0,0]

def message(A,B):
    return ((A*B)/(A*B).sum(axis=1).sum(axis=0)[0,0]).sum(axis=0)
    
print(message(x4_x2,x2_x1))
print(message(x2_x1,x2_x1))

