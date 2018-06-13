# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import numpy as np 
from itertools import combinations
import math
import pdb

"""
Parameter descriptions 

evaluation budget: number of treatment combinations to evaluate
treatment_budget: size of treated state subset
state_scores: list of goodness-scores associated with each state
"""


def nCk(n, r):
  f = math.factorial
  return f(n) / f(r) / f(n - r)


def num_candidate_states(evaluation_budget, treatment_budget):
  """
  :return num_candidates: number of actions we can afford to evaluate under these budget
  """
  num_candidates = treatment_budget
  while nCk(num_candidates, treatment_budget) <= evaluation_budget:
    num_candidates += 1
  return num_candidates - 1


def all_candidate_actions(state_scores, evaluation_budget, treatment_budget):
  """
  :return candidate_actions: list of candidate actions according to state_scores
  """
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


def average_Q_over_random_actions(Q_fn, treatment_budget, L):
  """
  Evaluate the Q_fn at random actions with treatment_budget
  treatments.
  """
  dummy_action = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
  random_Qs = []
  for i in range(20):
    action = np.random.permutation(dummy_action)
    random_Qs.append(Q_fn(action))
  return np.mean(np.array(random_Qs), axis=0)


def Q_max(Q_fn, evaluation_budget, treatment_budget, L):
  """
  :return best_q: q-value associated with best candidate action
  """
  # state_scores = Q_fn(np.ones(L))
  # state_scores = average_Q_over_random_actions(Q_fn, treatment_budget, L)
  # actions = all_candidate_actions(state_scores, evaluation_budget, treatment_budget)

  all_action_indices = combinations(range(16), 3)
  best_q = float('inf')
  q_vals = []
  # for i in range(actions.shape[0]):
  for ixs in all_action_indices:
    # a = actions[i,:]
    a = np.zeros(L)
    a[list(ixs)] = 1
    q = Q_fn(a)
    # print('q: {}'.format(q))
    if np.sum(q) < np.sum(best_q):
      best_q = q 
      best_a = a
    q_vals.append(np.sum(q))

  # Random actions for debugging
  # dummy_action = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
  # random_Qs = np.array([np.sum(Q_fn(np.random.permutation(dummy_action))) for i in range(84)])
  # print('best candidate: {} best random: {} worst random: {}'.format(np.sum(best_q),
  #                                                                    np.min(random_Qs), np.max(random_Qs)))
  return best_q, best_a, q_vals


def Q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model):
  """
  :return best_q_arr: array of max q values associated with each state in state_score_history
  """
  # Q = lambda s: Q_max(Q_fn, s, evaluation_budget, treatment_budget)
  best_q_arr = []
  argmax_data_blocks = []
  argmax_actions = []
  for t in range(env.T):
    Q_fn_t = lambda a: Q(a, env.X_raw[t], env, predictive_model)
    Q_max_t, Q_argmax_t, q_vals = Q_max(Q_fn_t, evaluation_budget, treatment_budget, env.L)
    best_q_arr.append(Q_max_t)
    best_data_block = env.data_block_at_action(env.X_raw[t], Q_argmax_t)
    argmax_data_blocks.append(best_data_block)
    argmax_actions.append(Q_argmax_t)
  return np.array(best_q_arr), argmax_data_blocks, argmax_actions, q_vals


def Q(a, raw_data_block, env, predictive_model, network_features=False):
  if network_features:
    data_block = env.network_features_at_action(raw_data_block, a)
  else:
    data_block = env.data_block_at_action(raw_data_block, a)
  Qhat = predictive_model(data_block)
  return Qhat

    

  
  



