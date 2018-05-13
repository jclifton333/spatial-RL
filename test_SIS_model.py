# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np
from generate_network import lattice
from SIS_model import SIS
from autologit import AutoRegressor, data_block_at_action
from lookahead import lookahead, Q_max
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from QL_objective import Qopt, Q_from_rollout_features

def main(K, L, T, nRep, method='QL', rollout_feature_times=[1]):
  '''
  :param K: lookahead depth
  :param L: number of locations in network
  :param T: duration of simulation rep
  :param nRep: number of replicates
  :param method: string in ['QL', 'rollout', 'random']
  '''
  #Initialize generative model
  omega = 1
  sigma = np.array([-1, -10, -1, -10, -10, 0, 0]) 
  gamma = 0.9
  m = lattice(L)
  g = SIS(m, omega, sigma)

  #Evaluation limit parameters 
  evaluation_budget = 10
  treatment_budget = 4

  #Initialize AR object
  #feature_function = lambda x: x
  poly = PolynomialFeatures(interaction_only=True)
  feature_function = lambda data_block: poly.fit_transform(data_block)
  
  AR = AutoRegressor(LogisticRegression(), LogisticRegression(), 
                     Ridge(), feature_function)

  means = []
  for rep in range(nRep):
    a = np.random.binomial(n=1, p=treatment_budget/L, size=L)
    g.reset()
    print('Rep: {}'.format(rep))
    for i in range(T):
      g.step(a)   
      if method == 'random':
        a = np.random.binomial(1, treatment_budget/L, size=L)
      else:
        Qargmax, rollout_feature_list, rollout_Q_function_list = lookahead(K, gamma, g, evaluation_budget, treatment_budget, AR, rollout_feature_times) 
        
        if method == 'QL':        
          thetaOpt = Qopt(rollout_feature_list, rollout_Q_function_list, gamma, 
                          g, evaluation_budget, treatment_budget)
          Q = lambda a: Q_from_rollout_features(data_block_at_action(g, i, a, None), thetaOpt, 
                                                rollout_feature_list, rollout_Q_function_list)
          _, a, _ = Q_max(Q, evaluation_budget, treatment_budget, L)          
        elif method == 'rollout':
          a = Qargmax
    means.append(np.sum(g.Y))
  return g, AR, means

_, _, f0 = main(3, 9, 20, 5, method='rollout', rollout_feature_times=[1,2])
_, _, f1 = main(6, 9, 20, 5, method='rollout', rollout_feature_times=[1,3,5])
_, _, f2 = main(8, 9, 20, 5, method='rollout', rollout_feature_times=[1,3,5,7])
_, _, f3 = main(10, 9, 20, 5, method='rollout', rollout_feature_times=[1,3,5,7,9])