# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import numpy as np 
import math
from autologit import AutoRegressor, data_block_at_action
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
import pdb

'''
Parameter descriptions 

evaluation budget: number of treatment combinations to evaluate
treatment_budget: size of treated state subset
state_scores: list of goodness-scores associated with each state
'''

def nCk(n, k):
  f = math.factorial
  return f(n) / f(k) / f(n - k)

def num_candidate_states(evaluation_budget, treatment_budget):
  '''
  :return num_candidates: number of actions we can afford to evaluate under these budget
  '''  
  num_candidates = treatment_budget
  while nCk(num_candidates, treatment_budget) <= evaluation_budget: 
    num_candidates += 1
  return num_candidates - 1

def all_candidate_actions(state_scores, evaluation_budget, treatment_budget):
  '''
  :return candidate_actions: list of candidate actions according to state_scores
  '''
  num_candidates = num_candidate_states(evaluation_budget, treatment_budget)
  sorted_indices = np.argsort(state_scores)
  candidate_indices = sorted_indices[:num_candidates]
  candidate_treatment_combinations = combinations(candidate_indices, treatment_budget)
  candidate_treatment_combinations = [list(combo) for combo in candidate_treatment_combinations]
  candidate_actions = np.zeros((0, len(state_scores)))
  for i in range(len(candidate_treatment_combinations)):
    a = np.zeros(len(state_scores))
    a[candidate_treatment_combinations[i]] = 1
    candidate_actions = np.vstack((candidate_actions, np.array(a)))
  return candidate_actions
  
def Q_max(Q_fn, state_scores, evaluation_budget, treatment_budget, nS):
  '''
  :return best_q: q-value associated with best candidate action
  '''
  Q_treat_all = Q_fn(np.ones(nS)) 
  actions = all_candidate_actions(Q_treat_all, evaluation_budget, treatment_budget)
  best_q = float('inf')  
  q_vals = []
  for i in range(actions.shape[0]):
    a = actions[i,:]
    q = Q_fn(a)
    if np.sum(q) < np.sum(best_q):
      best_q = q 
      best_a = a
    q_vals.append(np.sum(q))
  return best_q, best_a, q_vals

def Q_max_all_states(model, evaluation_budget, treatment_budget, predictive_model, feature_function, predicted_probs_list = None):
  '''
  :return best_q_arr: array of max q values associated with each state in state_score_history
  '''
  #Q = lambda s: Q_max(Q_fn, s, evaluation_budget, treatment_budget)
  best_q_arr = np.array([])
  for t in range(model.T):
    if predicted_probs_list is None:
      Q_fn_t = lambda a: Q(a, predictive_model, model, t, feature_function, predicted_probs = None)
    else:
      Q_fn_t = lambda a: Q(a, predictive_model, model, t, feature_function, predicted_probs = None)
    Q_max_t, Q_argmax_t, q_vals = Q_max(Q_fn_t, model.S[t,:], evaluation_budget, treatment_budget, model.nS)
    best_q_arr = np.append(best_q_arr, Q_max_t)
  return best_q_arr, Q_argmax_t, q_vals

def Q(a, predictive_model, model, t, feature_function, predicted_probs):
  # Add a to data 
  data_block = data_block_at_action(model, t, a, feature_function, predicted_probs)  
  predicted_probs = predictive_model(data_block)
  return predicted_probs

def lookahead(K, gamma, env, evaluation_budget, treatment_budget, AR):
  AR.createAutoregressionDataset(env)
  target = np.hstack(env.y).astype(float)
  
  #Fit 1-step model
  AR.fitClassifier(target)
  Qmax, Qargmax, qvals = Q_max_all_states(env, evaluation_budget, treatment_budget, AR.autologitPredictor, AR.featureFunction, AR.pHat_uc)
  
  #Look ahead 
  for k in range(K-1):
    target += gamma*Qmax
    AR.fitRegressor(target)
    if k < K-2:
      Qmax, Qargmax, qvals = Q_max_all_states(env, evaluation_budget, treatment_budget, AR.autologitPredictor, AR.featureFunction, AR.pHat_uc)
  return Qargmax
    

  
  



