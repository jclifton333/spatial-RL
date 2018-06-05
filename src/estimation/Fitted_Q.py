# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:08:33 2018

@author: Jesse
"""
import numpy as np
from .Q_functions import Q_max, Q_max_all_states
from scipy.special import expit, logit
import pdb

def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features


def rollout(K, gamma, env, evaluation_budget, treatment_budget, AR, rollout_feature_times):
  AR.resetPredictors()
  
  target = np.hstack(env.y).astype(float)
  # target = np.hstack(env.true_infection_probs).astype(float)
  rollout_feature_list = []
  
  #Fit 1-step model
  AR.fitClassifier(env, target, True)
  one_step_predictions = expit(AR.autologitPredictor(env.X[-1]))
  true_probs = env.true_infection_probs[-1]
  r2 = np.mean(-true_probs * np.log(one_step_predictions) - (1 - true_probs) * np.log(1 - one_step_predictions))
  Qmax, Qargmax, argmax_actions, qvals = Q_max_all_states(env, evaluation_budget, treatment_budget,  AR.autologitPredictor)
  #Look ahead
  for k in range(1, K):
    target += gamma*Qmax.flatten()
    rollout_feature_time = k in rollout_feature_times
    AR.fitRegressor(env, target, rollout_feature_time)
    if rollout_feature_time:
      Q_features_at_each_block = [np.sum(AR.autologitPredictor(env.X[t])) for t in range(len(env.X))]
      rollout_feature_list.append(Q_features_at_each_block)
    Qmax, Qargmax, argmax_actions, qvals = Q_max_all_states(env, evaluation_budget, treatment_budget, AR.autologitPredictor)
  return argmax_actions, rollout_feature_list, AR.predictors, target, r2
