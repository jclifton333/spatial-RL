# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:08:33 2018

@author: Jesse
"""
import numpy as np
from src.estimation.optim.q_max import q_max_all_states
from scipy.special import expit, logit
import pdb
import time


def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features


def rollout(K, gamma, env, evaluation_budget, treatment_budget, regressor, argmaxer):
  regressor.resetPredictors()
  target = np.hstack(env.y).astype(float)

  # Fit 1-step model
  regressor.fitClassifier(env, target, True)
  q_max, a = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor, argmaxer)

  # Look ahead
  for k in range(1, K):
    target += gamma*q_max.flatten()
    regressor.fitRegressor(env, target, False)
    q_max, a = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor, argmaxer)
  return a


# def network_features_rollout(env, evaluation_budget, treatment_budget, regressor):
#   # target = np.sum(env.y, axis=1).astype(float)
#   target = np.sum(env.true_infection_probs, axis=1).astype(float)
#   regressor.fit(np.array(env.Phi), target)
#   Qmax, Qargmax, argmax_actions, qvals = q_max(env, evaluation_budget, treatment_budget, regressor.predict, network_features=True)
#   return argmax_actions, target
