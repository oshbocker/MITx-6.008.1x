# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:04:35 2016

@author: mysterion
"""

import numpy as np

psi = np.array([[3/4,1/4],[1/4,3/4]])
phi_1 = np.array([1/2*1/2,1/2*1/4])
phi_2 = np.array([1/2,1/4])
phi_3 = np.array([1/2,3/4])
phi_4 = np.array([1/2,3/4])
phi_5 = np.array([1/2,3/4])
phi = np.array([phi_1,phi_2,phi_3,phi_4,phi_5])

gamma = np.log2(3)

psi = -np.log2(psi)
phi = -np.log2(phi)

messages = np.zeros((2,5))
tracebacks = np.zeros((2,4))

m1_2 = np.zeros(2)
t2_1 = np.zeros(2)

m1_2[0] = np.amin(phi[0]+psi[:,0])
t2_1[0] = np.argmin(phi[0]+psi[:,0])

m1_2[1] = np.amin(phi[0]+psi[:,1])
t2_1[1] = np.argmin(phi[0]+psi[:,1])

m2_3 = np.zeros(2)
t3_2 = np.zeros(2)

m2_3[0] = np.amin(phi[1]+psi[:,0]+m1_2)
t3_2[0] = np.argmin(phi[1]+psi[:,0]+m1_2)

m2_3[1] = np.amin(phi[1]+psi[:,1]+m1_2)
t3_2[1] = np.argmin(phi[1]+psi[:,1]+m1_2)

for i in range(1,5):
    new_matrix = phi[i-1] + psi + messages[:,i-1]
    messages[:,i] = np.amin(new_matrix, axis=1)
    tracebacks[:,i-1] = np.argmin(new_matrix, axis=1)

x5_hat = np.argmin(messages[:,-1]+phi[4])
viterb = np.zeros(5)
viterb[-1] = x5_hat
for i in reversed(range(0,4)):
    viterb[i] = tracebacks[int(viterb[i+1]),i]
