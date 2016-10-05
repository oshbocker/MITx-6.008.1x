# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:38:14 2016

@author: mysterion
"""
import numpy as np

# Entropy
def entropy(prob_vector):
    return sum([x*np.log2(1/x) for x in prob_vector])
    
L1 = [999999/1000000,1/1000000]
L2 = [999999/1000000,1/1000000]
L3 = [9/10,1/10]
L1_entropy = entropy(L1)
L2_entropy = entropy(L2)
L3_entropy = entropy(L3)
print("L1 entropy: ", L1_entropy)
print("L2 entropy: ", L2_entropy)
print("L3 entropy: ", L3_entropy)


probs = np.arange(0,1,0.01)
entropy_list = []
for p in probs:
    entropy_list.append(entropy([p,1-p]))
    
import matplotlib.pyplot as plt

plt.scatter(probs,entropy_list)

x = np.arange(0,1,0.01)
plt.figure()
plt.plot(x, np.log(x))
plt.plot(x, x - 1)
plt.xlabel('x')
plt.legend(['ln(x)', 'x-1'], loc=4)
plt.show()

# Mutual Information
joint_prob_XY = np.array([[0.10, 0.09, 0.11], [0.08, 0.07, 0.07], [0.18, 0.13, 0.17]])

# marginal distributions
prob_X = joint_prob_XY.sum(axis=1)
prob_Y = joint_prob_XY.sum(axis=0)

# the joint probability table if X and Y were actually independent
joint_prob_XY_indep = np.outer(prob_X, prob_Y)
print(joint_prob_XY)
print(joint_prob_XY_indep)

def information_divergence(p,q):
    if p.shape != q.shape:
        raise ValueError("p and q distributions must have the same shape")
    divergence_sum = 0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            divergence_sum += p[x][y]*np.log2(p[x][y]/q[x][y])
    return divergence_sum
            
# With NumPy, we can code the information diverence as a one-liner
info_divergence = lambda p, q: np.sum(p * np.log2(p/q))

print(information_divergence(joint_prob_XY,joint_prob_XY_indep))
print(info_divergence(joint_prob_XY,joint_prob_XY_indep))

# Ainsley Works on Problem Sets

cs = np.arange(1,5,1)
print("One: ", (1/3)*np.log2(3))
cs = 1/(2*cs + 1)*np.log2(1/(1/(2*cs + 1)))
print("cs: ",cs, "\n\n")

import scipy.special

def c_given_d(q,d):
    S = np.arange(1,5,1)
    exp_c_given_s = np.arange(1,5,1)
    denominator = sum([scipy.special.binom(4,x)*q**d*(1-q)**(x-d) for x in S[d:]])
    prob_S_given_D = [scipy.special.binom(4,x)*q**d*(1-q)**(x-d) if x >= d else 0. for x in S]/denominator
    prob_S_given_D_norm = prob_S_given_D/sum(prob_S_given_D)
    return sum(exp_c_given_s*prob_S_given_D_norm), prob_S_given_D_norm

print("q=0.2, d=1: ", c_given_d(0.2,1), "\n\n")
print("q=0.5, d=2: ", c_given_d(0.5,2), "\n\n")
print("q=0.7, d=3: ", c_given_d(0.7,3), "\n\n")
print("q=0.2, d=3: ", c_given_d(0.01,3))

