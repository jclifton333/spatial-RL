# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:08:33 2018

@author: Jesse
"""
import numpy as np
from src.estimation.q_functions.q_max import q_max_all_states


def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features


def rollout(K, gamma, env, evaluation_budget, treatment_budget, regressor, argmaxer, bootstrap=True, ixs=None):
  if ixs is None:
    target = np.hstack(env.y).astype(float)
    features = np.vstack(env.X)
  else:
    target = np.hstack([env.y[i][ixs[i]] for i in range(len(env.y))])
    features = np.vstack([env.X[i][ixs[i]] for i in range(len(env.X))])

  if bootstrap:
    weights = np.random.exponential(size=len(target))
  else:
    weights = None
  # Fit 1-step model
  regressor.fitClassifier(features, target, weights, True, env.add_neighbor_sums)
  q_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor, argmaxer, ixs)
  # Look ahead
  for k in range(1, K + 1):
    target += gamma*q_max.flatten()
    regressor.fitRegressor(features, target, weights, False)
    if k < K:
      q_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, regressor.autologitPredictor,
                                     argmaxer, ixs)
  return regressor.autologitPredictor


def rollout_variance_estimate(K, gamma, env, evaluation_budget, treatmnet_budget, regressor, argmaxer,
                              infection_probabilities, num_rep=100):
  """

  :param K:
  :param gamma:
  :param env:
  :param evaluation_budget:
  :param treatmnet_budget:
  :param regressor:
  :param argmaxer:
  :param infection_probabilities: Vector of estimated infection probabilities, for parametric bootstrapping.
  :param num_rep:
  :return:
  """
  y_tilde = np.random.binomial(1, p=infection_probabilities)
  for rep in range(num_rep):
    # q0 ~ y_tilde
    # q1 ~ y_tilde + \gamma max q0
    # q1_tilde = bootstrap(q1)
    # back up until K



# def network_features_rollout(env, evaluation_budget, treatment_budget, regressor):
#   # target = np.sum(env.y, axis=1).astype(float)
#   target = np.sum(env.true_infection_probs, axis=1).astype(float)
#   regressor.fit(np.array(env.Phi), target)
#   Qmax, Qargmax, argmax_actions, qvals = q_max(env, evaluation_budget, treatment_budget, regressor.predict, network_features=True)
#   return argmax_actions, target
