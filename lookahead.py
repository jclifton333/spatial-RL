# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import numpy as np 
import math
from autologit import autologit 
from scipy.optimize import fmin_tnc
from itertools import combinations


def nCk(n, k):
  f = math.factorial
  return f(n) / f(k) / f(n - k)

def num_candidate_states(evaluation_budget, treatment_budget):
  num_candidates = treatment_budget
  while nCk(num_candidates, treatment_budget) <= evaluation_budget: 
    num_candidates += 1
  return num_candidates - 1

def all_candidate_actions(state_scores, evaluation_budget, treatment_budget):
  num_candidates = num_candidate_states(evaluation_budget, treatment_budget)
  sorted_indices = np.argsort(-state_scores)
  candidate_indices = sorted_indices[:num_candidates]
  candidate_treatment_combinations = combinations(candidate_indices, treatment_budget)
  candidate_treatment_combinations = [list(combo) for combo in candidate_treatment_combinations]
  candidate_actions = np.zeros((0, len(state_scores)))
  for i in range(len(candidate_treatment_combinations)):
    a = np.zeros(len(state_scores))
    a[candidate_treatment_combinations[i]] = 1
    candidate_actions = np.vstack((candidate_actions, np.array(a)))
  
def Q_max(Q_fn, state_scores, evaluation_budget, treatment_budget):
  actions = all_candidate_actions(state_scores, evaluation_budget, treatment_budget)
  best_q = -float('inf')    
  for i in range(actions.shape[0]):
    a = actions[i,:]
    q = Q_fn(a)
    if -Q_fn > best_q:
      best_q = q 
  return best_q

def Q_max_all_states(Q_fn, state_score_history, evaluation_budget, treatment_budget):
  Q = lambda s: Q_max(Q_fn, s, evaluation_budget, treatment_budget)
  q_max = np.array([Q(state_score_history[i,:]) for i in range(state_score_history.shape[0])])
  return q_max

def lookahead(k, model, evaluation_budget, treatment_budget, autologit_classifier, unconditional_classifier):
  pass
  
  



