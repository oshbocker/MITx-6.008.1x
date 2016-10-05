# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 09:57:04 2016

@author: mysterion
"""

import random
import numpy as np

# Expected value
sum_of_dice = 0.

#for i in range(10000):
#    sum_of_dice += random.choice([1,2,3,4,5,6])

#print(sum_of_dice/10000.)

# Variance

exp_L1 = -1*(999999/1000000) + 999*(1/1000000)
print(exp_L1)

exp_L2 = -1*(999999/1000000) + 999999*(1/1000000)
print(exp_L2)

exp_L3 = -1*(9/10) + 9*(1/10)
print(exp_L3)

var_L1 = (-1-exp_L1)**2*(999999/1000000) + (999-exp_L1)**2*(1/1000000)
print(var_L1)

var_L2 = (-1-exp_L2)**2*(999999/1000000) + (999999-exp_L2)**2*(1/1000000)
print(var_L2)

var_L3 = (-1-exp_L3)**2*(9/10) + (9-exp_L3)**2*(1/10)
print(var_L3)
print(np.sqrt(var_L1))
print(np.sqrt(var_L2))
print(np.sqrt(var_L3))

# Medical Diagnosis
a = 0.09016
b = 0.00001

exp_pos_not_treat = 50000*b
exp_pos_treat = 20000*b + 20000*(1-b)
print(exp_pos_not_treat,exp_pos_treat)

# Expectations of Multiple Random Variables
caesar = (1/26)**6
print(caesar)
print("caesar: ",caesar*(1000000000-5))

king_exp = 51*(4/52)*(3/51)
print(king_exp)

deck = np.zeros(52)
deck[:4] = 1
tot_king_pairs = 0

for i in range(1):
    np.random.shuffle(deck)
    king_pairs = 0
    last_card = deck[0]
    for card in deck[1:]:
        if card + last_card == 2.:
            king_pairs += 1
        last_card = card
    tot_king_pairs += king_pairs
            
print(tot_king_pairs/100000)
        
