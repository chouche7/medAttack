#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 09:48:58 2023

@author: chouche7
"""
#read advanced metric
import pickle
import numpy as np
import matplotlib.pyplot as plt



#create the list of L1 regularization terms
seq = np.linspace(np.log(0.0005), np.log(0.015), num=10)
lamb_list = np.exp(seq)


plt.figure()
pp = []
sr = []

avg_avg = []
avg_max = []
for i in range(5):
    path_string = '../../mimic3/fold_' + str(i) + '/output/Adv_metric.pkl'
    fn = open(path_string, 'rb')
    results = pickle.load(fn)
    
    #[0] maximum perturbation
    max_perturb = results[0]
    avg_max.append(np.mean(max_perturb, axis = 0))
    

    #[1] average perturbation
    avg_perturb = results[1]
    avg_avg.append(np.mean(avg_perturb, axis = 0))
    #[2] number of location changed
    loc_change = results[2]
    #perturbation percentage (19 is the number of features, 48 is the number of observations per input)
    pp.append(np.mean(loc_change/(19*48), axis = 0))
    
    #[3] flags if classification change due to adversarial attack
    flags = results[3]
    sr.append(np.sum(flags, axis = 0)/max_perturb.shape[0])
    
    loc = np.max(loc_change, axis = 0)
    plt.plot(lamb_list, loc, label = "Fold " + str(i))

plt.title("Magnitude vs Sparsity")
plt.xlabel("Regularization")
plt.ylabel("Perturbed")  
plt.legend(prop={'size': 8}) 
plt.show()

pp = np.array(pp).T
sr = np.array(sr).T

avg_pp = np.mean(pp, axis = 1)
avg_sr = np.mean(sr, axis = 1)

plt.figure()
plt.plot(pp, sr, label = ["Fold 0", "Fold 1","Fold 2","Fold 3","Fold 4"])
plt.plot(avg_pp, avg_sr, label = "Avg", linestyle='dashed')
plt.title("Susceptibility")
plt.xlabel("Perturbed (%)")
plt.ylabel("Success Rate")  
plt.legend(prop={'size': 8}) 
plt.show()


avg_max = np.mean(np.array(avg_max).T, axis = 1)
avg_avg = np.mean(np.array(avg_avg).T, axis = 1)

plt.figure()
plt.scatter(avg_max, avg_sr)
plt.title("Adversarial Perturbation Assessment")
plt.xlabel("Max Perturbation")
plt.ylabel("Success Rate")  
plt.show()

# plt.figure()
# plt.scatter(avg_avg, avg_sr)
# plt.title("Adversarial Perturbation Assessment")
# plt.xlabel("Avg Perturbation")
# plt.ylabel("Success Rate")  
# plt.show()

