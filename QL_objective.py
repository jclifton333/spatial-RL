# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse
"""

import numpy as np
from lookahead import Q_max_all_states
from scipy.optimize import minimize

def Q_from_rollout_features(data_block, theta, rollout_feature_list, rollout_Q_function_list):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list])
  return np.dot(rollout_Q_features, theta)
     
def QL_objective(theta, rollout_feature_list, rollout_Q_function_list, gamma, env, evaluation_budget, 
                 treatment_budget, feature_function):   
  Q_fn = lambda data_block: Q_from_rollout_features(data_block, theta, rollout_feature_list, rollout_Q_function_list)
  Q = np.array([Q_fn(data_block) for data_block in env.X])
  Qmax, _, _ = Q_max_all_states(env, evaluation_budget, treatment_budget, Q_fn, feature_function)
  TD = env.R + gamma*Qmax - Q
  return np.dot(TD, TD)

def Qopt(rollout_feature_list, rollout_Q_function_list, gamma, env, evaluation_budget, 
                 treatment_budget, feature_function):
  objective = lambda theta: QL_objective(theta, rollout_feature_list, rollout_Q_function_list, gamma, env, evaluation_budget, 
                 treatment_budget, feature_function)
  soln = minimize(objective, x0=np.zeros(len(rollout_Q_function_list)), method='L-BFGS-B')
  return soln.x