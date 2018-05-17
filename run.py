# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np
from generate_network import lattice
from SIS_model import SIS
from autologit import AutoRegressor
from lookahead import lookahead, Q_max
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from features import polynomialFeatures
from QL_objective import GGQ, rollout_Q_features
import pdb


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
  featureFunction = polynomialFeatures(3, interaction_only=True)  
  #featureFunction = lambda d: d
  g = SIS(m, omega, sigma, featureFunction)

  #Evaluation limit parameters 
  treatment_budget = 20
  evaluation_budget = 1

  #Initialize AR object
  #feature_function = lambda x: x
  AR = AutoRegressor(LogisticRegression, Ridge)

  means = []
  a_dummy = np.append(np.ones(treatment_budget), np.zeros(L - treatment_budget))
  for rep in range(nRep):
    #print('Rep: {}'.format(rep))
    g.reset()
    a = np.random.permutation(a_dummy)
    g.step(a)
    a = np.random.permutation(a_dummy)
    for i in range(T):
      #print('i: {}'.format(i))
      g.step(a)   
      if method == 'random':
        a = np.random.permutation(a_dummy)
        target = None
      else:
        argmax_actions, rollout_feature_list, rollout_Q_function_list, target = lookahead(K, gamma, g, evaluation_budget, 
                                                                           treatment_budget, AR, rollout_feature_times)     
        if method == 'QL':        
          thetaOpt = GGQ(rollout_feature_list, rollout_Q_function_list, gamma, 
                          g, evaluation_budget, treatment_budget, True)
          Q = lambda a: np.dot(rollout_Q_features(g.data_block_at_action(g.X_raw[i], a), rollout_Q_function_list, intercept=True),
                               thetaOpt)
          _, a, _ = Q_max(Q, evaluation_budget, treatment_budget, L)          
        elif method == 'rollout':
          a = argmax_actions[-1]
    means.append(np.mean(g.Y))
  return g, AR, means, target

if __name__ == '__main__':
  import time
  for k in range(0,10):
    t0 = time.time()
    _, _, scores, _ = main(k, 100, 25, 10, method='random', rollout_feature_times=[])
    t1 = time.time()
    print('k={}: score={} se={} time={}'.format(k, np.mean(scores), np.std(scores) / np.sqrt(100), t1 - t0))
