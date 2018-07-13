# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:08:33 2018

@author: Jesse
"""
import numpy as np
from src.estimation.optim.q_max import q_max_all_states
from src.estimation.q_functions.q_functions import q
from functools import partial
from scipy.special import expit, logit
import pdb
import time


def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features


def rollout(K, gamma, env, evaluation_budget, treatment_budget, regressor, argmaxer, ixs=None):
  regressor.resetPredictors()
  if ixs is None:
    target = np.hstack(env.y).astype(float)
    features = np.vstack(env.X)
  else:
    try:
      target = np.hstack([env.y[i][ixs[i]] for i in range(len(env.y))])
    except:
      pdb.set_trace()
    features = np.vstack([env.X[i][ixs[i]] for i in range(len(env.X))])

  # Fit 1-step model
  regressor.fitClassifier(features, target, True)
  q_max = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor, argmaxer, ixs)

  # Look ahead
  for k in range(1, K):
    target += gamma*q_max.flatten()
    regressor.fitRegressor(features, target, False)
    if k < K-1:
      q_max = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor, argmaxer, ixs)
  return regressor.autologitPredictor


# def network_features_rollout(env, evaluation_budget, treatment_budget, regressor):
#   # target = np.sum(env.y, axis=1).astype(float)
#   target = np.sum(env.true_infection_probs, axis=1).astype(float)
#   regressor.fit(np.array(env.Phi), target)
#   Qmax, Qargmax, argmax_actions, qvals = q_max(env, evaluation_budget, treatment_budget, regressor.predict, network_features=True)
#   return argmax_actions, target
