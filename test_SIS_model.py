# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np
from generate_network import lattice
from SIS_model import SIS
from autologit import AutoRegressor
from lookahead import lookahead
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from QL_objective import Qopt

def main(K, L, T, nRep, random=False):
  #Initialize generative model
  omega = 1
  sigma = np.array([-1, -10, -1, -10, -10, 0, 0]) 
  gamma = 0.9
  m = lattice(L)
  g = SIS(m, omega, sigma)
  rollout_feature_times = [1,2]

  #Evaluation limit parameters 
  evaluation_budget = 20
  treatment_budget = 15

  #Initialize AR object
  feature_function = lambda x: x
  AR = AutoRegressor(RandomForestClassifier(n_estimators=30), RandomForestClassifier(n_estimators=30), 
                     RandomForestRegressor(n_estimators=30), feature_function)

  means = []
  for rep in range(nRep):
    a = np.random.binomial(n=1, p=treatment_budget/L, size=L)
    g.reset()
    print('Rep: {}'.format(rep))
    for i in range(T):
      g.step(a)   
      if random:
        a = np.random.binomial(1, treatment_budget/L, size=L)
      else:
        Qargmax, rollout_feature_list, rollout_Q_function_list = lookahead(K, gamma, g, evaluation_budget, treatment_budget, AR, rollout_feature_times) 
        thetaOpt = Qopt(rollout_feature_list, rollout_Q_function_list, gamma, g, evaluation_budget, 
                        treatment_budget, feature_function)
      means.append(np.sum(g.Y))
  return g, AR, means

_, _, m0 = main(3, 20, 20, 5)
