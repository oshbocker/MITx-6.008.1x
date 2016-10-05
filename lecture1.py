# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import comp_prob_inference

prob_space = {'sunny': 1/2, 'rainy': 1/6, 'snowy': 1/3}
random_outcome = comp_prob_inference.sample_from_finite_probability_space(prob_space)

# Aproach 1
W_mapping = {'sunny': 'sunny', 'rainy': 'rainy', 'snowy': 'snowy'}
I_mapping = {'sunny': 1, 'rainy': 0, 'snowy': 0}

W1 = W_mapping[random_outcome]
I1 = I_mapping[random_outcome]

# Approach 2
W_table = {'sunny': 1/2, 'rainy': 1/6, 'snowy': 1/3}
I_table = {0: 1/2, 1: 1/2}

W2 = comp_prob_inference.sample_from_finite_probability_space(W_table)
I2 = comp_prob_inference.sample_from_finite_probability_space(I_table)

print("W1 :", W1, "I1: ", I1)
print("W2 :", W2, "I2: ", I2)