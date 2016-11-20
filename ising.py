# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:50:41 2016

@author: mysterion
"""
## Homework: Ising Model
import numpy as np
from matplotlib import pyplot as plt

poles = [1,-1]
ising = []

for x1 in poles:
    for x2 in poles:
        for x3 in poles:
            ising.append((x1,x2,x3))


def model(x1,x2,x3):
    return x1*x2+x2*x3

def ising_prob(alpha):
    poles = [1,-1]
    ising = []

    for x1 in poles:
        for x2 in poles:
            for x3 in poles:
                ising.append((x1,x2,x3))
    def model(x1,x2,x3):           
        Z = 2*np.exp(1)**(2*alpha)+2*np.exp(1)**(-2*alpha)+4
        return np.log((np.exp(1)**(alpha*x1*x2)*np.exp(1)**(alpha*x2*x3))/Z)
    
    ising_dict = {}
    
    for i in ising:
        if i not in ising_dict:
            ising_dict[i] = [model(i[0],i[1],i[2])]
    
    return ising_dict
    
def ising_exp(alpha):
    ising_dict = ising_prob(alpha)
    
    pole_dict = {2:np.array(ising_dict[(1,1,1)])+np.array(ising_dict[(-1,-1,-1)]), \
    0:np.array(ising_dict[(-1,1,1)])+np.array(ising_dict[(-1,-1,1)])+ \
    np.array(ising_dict[(1,-1,-1)])+np.array(ising_dict[(1,1,-1)]), \
    -2:np.array(ising_dict[(1,-1,1)])+np.array(ising_dict[(-1,1,-1)])}
    return pole_dict

def g(m,alpha,n):
    return 2*m*n*alpha-n*(np.log(2*np.exp(1)**(2*alpha)+2*np.exp(1)**(-2*alpha)+4))

print(g(0.2,0.20273255,17))    
print(g(0.9,1.4722195,15))
