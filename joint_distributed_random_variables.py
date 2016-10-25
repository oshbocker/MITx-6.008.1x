Ed# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import comp_prob_inference

# Approach 0: Don't actually represent the joint probability table
print("Approach 0\n")
prob_table = {('sunny', 'hot'): 3/10,
     ('sunny', 'cold'): 1/5,
     ('rainy', 'hot'): 1/30,
     ('rainy', 'cold'): 2/15,
     ('snowy', 'hot'): 0,
     ('snowy', 'cold'): 1/3}

print(prob_table[('rainy', 'cold')])

# Appraoch 1: Use dictionaries within a dictionary
print("\nApproach 1\n")
prob_W_T_dict = {}
for w in {'sunny', 'rainy', 'snowy'}:
    prob_W_T_dict[w] = {}
    
prob_W_T_dict['sunny']['hot'] = 3/10
prob_W_T_dict['sunny']['cold'] = 1/5
prob_W_T_dict['rainy']['hot'] = 1/30
prob_W_T_dict['rainy']['cold'] = 2/15
prob_W_T_dict['snowy']['hot'] = 0
prob_W_T_dict['snowy']['cold'] = 1/3

comp_prob_inference.print_joint_prob_table_dict(prob_W_T_dict)

# Approach 2: Use a 2D array
print("\nApproach 2\n")
prob_W_T_rows = ['sunny', 'rainy', 'snowy']
prob_W_T_cols = ['hot', 'cold']
prob_W_T_array = np.array([[3/10, 1/5], [1/30, 2/15], [0, 1/3]])
comp_prob_inference.print_joint_prob_table_array(prob_W_T_array, prob_W_T_rows, prob_W_T_cols)

prob_W_T_array[prob_W_T_rows.index('rainy'), prob_W_T_cols.index('cold')]

prob_W_T_row_mapping = {}
for index, label in enumerate(prob_W_T_rows):
    prob_W_T_row_mapping[label] = index
    
print(list(enumerate(prob_W_T_rows)))

prob_W_T_row_mapping = {label: index for index, label in enumerate(prob_W_T_rows)}
prob_W_T_col_mapping = {label: index for index, label in enumerate(prob_W_T_cols)}

# Exercise: Marginalization
print("\nMarginalization\n")
prob_table_X = {'sunny':1/2, 'rainy':1/6, 'snowy':1/3}
prob_table_Y = {1:1/2,0:1/2}
prob_table_X_Y = {'sunny':{1:1/4,0:1/4},'rainy':{1:1/12,0:1/12},'snowy':{1:1/6,0:1/6}}

# Exercise: Simpson's Paradox
print("\nSimpson's Paradox\n")
from simpsons_paradox_data import *

print(joint_prob_table[gender_mapping['female'], department_mapping['C'], admission_mapping['admitted']])

joint_prob_gender_admission = joint_prob_table.sum(axis=1)
print(joint_prob_gender_admission[gender_mapping['female'], admission_mapping['admitted']])

print("\nProbability Table conditioned on Female")
female_only = joint_prob_gender_admission[gender_mapping['female']]
prob_admission_given_female = female_only / np.sum(female_only)
prob_admission_given_female_dict = dict(zip(admission_labels, prob_admission_given_female))
print(prob_admission_given_female_dict)

print("\nProbability Table conditioned on Male")
male_only = joint_prob_gender_admission[gender_mapping['male']]
prob_admission_given_male = male_only / np.sum(male_only)
prob_admission_given_male_dict = dict(zip(admission_labels, prob_admission_given_male))
print(prob_admission_given_male_dict)

print("\nProbability Table conditioned on Admitted")
admitted_only = joint_prob_gender_admission[:, admission_mapping['admitted']]
prob_gender_given_admitted = admitted_only / np.sum(admitted_only)
prob_gender_given_admitted_dict = dict(zip(gender_labels, prob_gender_given_admitted))
print(prob_gender_given_admitted_dict)

print("\nWhich departments favor men?")
print("\nDept A - Female")
female_and_A_only = joint_prob_table[gender_mapping['female'], department_mapping['A']]
prob_admitted_given_female_and_A = female_and_A_only / np.sum(female_and_A_only)
prob_admitted_given_female_and_A_dict = dict(zip(admission_labels, prob_admitted_given_female_and_A))
print(prob_admitted_given_female_and_A_dict)

def department_discrimination(gender, dept):
    joint_prob_table_gender_dept = joint_prob_table[gender_mapping[gender], department_mapping[dept]]
    prob_admitted = joint_prob_table_gender_dept / np.sum(joint_prob_table_gender_dept)
    prob_admitted_dict = dict(zip(admission_labels,prob_admitted))
    return prob_admitted_dict
    
for gender in gender_labels:
    for dept in department_labels:
        print(str(dept),"-",str(gender),department_discrimination(gender,dept)['admitted'])
