# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:04:14 2016

@author: mysterion
"""

## Exercise: Twitter Follower Network
import numpy as np

observations = np.array([[1,1,1,0,0], \
                        [1,1,0,1,0], \
                        [1,1,1,1,1], \
                        [1,0,1,0,0], \
                        [1,1,1,0,0], \
                        [1,0,1,0,0], \
                        [0,0,0,0,0]])

theta_1 = sum(observations[:,0])/len(observations[:,0])
print("theta_1: ", theta_1)
theta_2_1 = sum(observations[:,1])/sum(observations[:,0])
print("theta_2_1: ", theta_2_1)
theta_3_1 = sum(observations[:,2])/sum(observations[:,0])
print("theta_3_1: ", theta_3_1)
theta_4_2 = sum(observations[:,3])/sum(observations[:,1])
print("theta_4_2: ", theta_4_2)
theta_5_2 = sum(observations[:,4])/sum(observations[:,1])
print("theta_5_2: ", theta_5_2)

print("\nUser Five Edge Likelihoods\n")
def user_five(i):
    
    return (sum(observations[:,i])-1)*np.log(3/4) + np.log(1/4)
    
for i in range(4):
    print(i+1, user_five(i))

## Exercise: The Chow-Liu Algorithm

mi = np.array([[0.3415,0.2845,0.0003,0.0822], \
               [0.2845,0.3457,0.0005,0.0726], \
               [0.0003,0.0005,0.5852,0.0002], \
               [0.0822,0.0726,0.0002,0.5948]])
               
## Homework: Alien Leaders

answers = np.array([[0,0,0], \
                    [0,0,1], \
                    [0,1,1], \
                    [1,1,1]])
                    

