import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from sklearn.linear_model import Ridge
from src.estimation.q_functions.model_fitters import SKLogit2
from src.estimation.q_functions.one_step import *
from src.environments.sis import convert_first_order_to_infection_status
from itertools import permutations, combinations
import matplotlib.pyplot as plt

import numpy as np


def sis_prediction_variance(X_new, XprimeX_inv):
  """
  Compute the variance of predictions at each state in X_new under the (linear) model-free predictor.

  :param X_new:
  :param XprimeX_inv: Here, 'X' is design matrix!
  :return:
  """
  # Compute prediction variances
  yhat_vars = np.array([])
  for x in X_new:
    var_yhat_x = np.dot(x.T, np.dot(XprimeX_inv, x))
    yhat_vars = np.append(yhat_vars, var_yhat_x)
  return yhat_vars


def sis_overlap(XA, XB):
  """
  Measure overlap between two networks with sis features.

  :param XA:
  :param XB:
  :return:
  """
  overlap = 0
  for xa, xb in zip(XA, XB):
    overlap += np.array_equal(xa, xb)
  return overlap


def sis_first_order_space_filler(env, number_of_neighbors, q_mb_one_step):
  """
  Space-filling design matrix for SIS model with first-order neighbor features.

  :param number_of_neighbors: Length-L vector of number of neighbors at each location.
  :param q_mb_one_step: model based prob estimator
  :return:
  """
  # ToDo: don't need both reps and sweeps anymore...
  NUM_REP = 100
  NUM_SWEEPS = 10
  QUANTILE_TO_KEEP = 0.95
  L = len(number_of_neighbors)
  # NUMBER_TO_KEEP_AT_EACH_SWEEP = int(np.floor(QUANTILE_TO_KEEP_AT_EACH_SWEEP * L))
  ALPHA = np.ones(8)
  dummy = np.array([1, 0, 0, 0, 0, 0, 0, 0])

  X_true = np.vstack(env.X)
  XprimeX = np.dot(X_true.T, X_true)
  XprimeX_inv = np.linalg.inv(XprimeX)
  vars_at_observed_states = sis_prediction_variance(X_true, XprimeX_inv)  # For comparison
  var_cutoff = np.quantile(vars_at_observed_states, QUANTILE_TO_KEEP)  # Use fake data for states exceeding this var

  X_synthetic = np.zeros((0, 16))
  y_synthetic = np.zeros(0)

  for sweep in range(NUM_SWEEPS*NUM_REP):
    # overlaps_for_sweep = []
    X_raw_sweep = np.random.binomial(1, 0.5, (L, 3))
    X_sweep = env.psi(X_raw_sweep, neighbor_order=1)
    p_sweep = q_mb_one_step(X_raw_sweep)
    y_sweep = np.random.binomial(1, p=p_sweep)

    vars_for_sweep = sis_prediction_variance(X_sweep, XprimeX_inv)
    locations_over_variance_threshold = np.where(vars_for_sweep > var_cutoff)
    X_synthetic = np.vstack((X_synthetic, X_sweep[locations_over_variance_threshold]))
    y_synthetic = np.hstack((y_synthetic, y_sweep[locations_over_variance_threshold]))

  print('num synthetic: {}'.format(len(y_synthetic)))
  return X_synthetic, y_synthetic


def sis_one_step_dyna_space_filling(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, gamma = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma']
  q_mb_one_step, _ = fit_one_step_sis_mb_q(env)

  MAX_NUM_NONZERO = int(np.min((env.max_number_of_neighbors, 8)))
  QUOTA = int(np.sqrt(env.L * env.T))
  infected_indices = [4, 5, 6, 7]

  X_synthetic, y_synthetic = sis_first_order_space_filler(env, env.adjacency_matrix.sum(axis=1), q_mb_one_step)
  X_new = np.vstack((np.vstack(env.X), X_synthetic))
  y_new = np.hstack((np.hstack(env.y), y_synthetic))
  infected_locations = np.where(X_new[:, infected_indices].sum(axis=1) == 1)
  q0 = SKLogit2()
  q0.fit(X_new, y_new, None, False, infected_locations, None)

  # Define q-function
  def qfn(a):
    x = env.data_block_at_action(-1, a)
    infected_locations = np.where(env.Y[-1, :] == 1)[0]
    return q0.predict_proba(x, infected_locations, None)

  a_ = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a_, {}


def sis_one_step_dyna(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, gamma = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma']
  q_mb_one_step, _ = fit_one_step_sis_mb_q(env)

  MAX_NUM_NONZERO = int(np.min((env.max_number_of_neighbors, 8)))
  QUOTA = int(np.sqrt(env.L * env.T))

  # raw_feature_combos = []
  # for i in range(8):
  #   dummy = [0]*8
  #   dummy[i] = 1
  #   raw_feature_combos.append(dummy)

  # # Build dictionary of feature indicator counts
  # # ToDo: this shouldn't be done each time the policy is called...
  # unique_feature_indicators = {}
  # for n in range(MAX_NUM_NONZERO):
  #   ixs = combinations(range(8), n+1)
  #   for ix in ixs:
  #     permutation_ = tuple([i for i in range(8) if i in ix])
  #     for raw_feature_combo in raw_feature_combos:
  #       feature_combo = tuple(raw_feature_combo) + permutation_
  #       unique_feature_indicators[feature_combo] = {'count': 0, 'list': []}

  mb_phats = [q_mb_one_step(x) for x in env.X]

  unique_feature_indicators = {}
  # Count number of feature indicators
  for t, X_ in enumerate(env.X):
    for l, x in enumerate(X_):
      x_indicator = x > 0
      if tuple(x_indicator) in unique_feature_indicators.keys():
        unique_feature_indicators[tuple(x_indicator)]['count'] += 1
        unique_feature_indicators[tuple(x_indicator)]['list'].append((x, t, l))
      else:
        unique_feature_indicators[tuple(x_indicator)] = {'count': 1, 'list': [(x, t, l)]}

  # # Add current state, infection info evaluated at treatment and no-treatment
  # X_current_trt = env.data_block_at_action(-1, np.zeros(env.L))
  # X_current_no_trt = env.data_block_at_action(-1, np.ones(env.L))
  # for l in range(env.L):
  #   x_current_trt_indicator = X_current_trt[l] > 0
  #   x_current_no_trt_indicator = X_current_no_trt[l] > 0
  #   if tuple(x_current_trt_indicator) in unique_feature_indicators.keys():
  #     unique_feature_indicators[tuple(x_current_trt_indicator)]['count'] += 1
  #     unique_feature_indicators[tuple(x_current_trt_indicator)]['list'].append((x_current_trt_indicator, env.T, l))
  #   else:
  #     unique_feature_indicators[tuple(x_current_trt_indicator)] = {'count': 1,
  #                                                                  'list': [(x_current_trt_indicator, env.T, l)]}
  #   if tuple(x_current_no_trt_indicator) in unique_feature_indicators.keys():
  #     unique_feature_indicators[tuple(x_current_no_trt_indicator)]['count'] += 1
  #     unique_feature_indicators[tuple(x_current_no_trt_indicator)]['list'].append((x_current_no_trt_indicator, env.T, l))
  #   else:
  #     unique_feature_indicators[tuple(x_current_no_trt_indicator)] = {'count': 1,
  #                                                                     'list': [(x_current_no_trt_indicator, env.T, l)]}

  # Supplement features that fall short of quota
  X_synthetic = np.zeros((0, env.X[0].shape[1]))
  Y_synthetic = np.zeros(0)
  # for feature_info in unique_feature_indicators.values():
  #   count = feature_info['count']
  #   if 0 < count < QUOTA:  # ToDo: what if count=0?
  #     num_fake_data = QUOTA - count

  #     # Sample with replacement up to desired number
  #     feature_list = feature_info['list']
  #     synthetic_ixs = np.random.choice(len(feature_list), num_fake_data, replace=True)
  #     synthetic = [feature_list[ix] for ix in synthetic_ixs]
  #     x_synthetic = [o_[0] for o_ in synthetic]
  #     y_synthetic = [mb_phats[o_[1]][o_[2]] for o_ in synthetic]
  #     # y_synthetic = np.random.binomial(1, p_synthetic)

  #     # Add to dataset
  #     X_synthetic = np.vstack((X_synthetic, x_synthetic))
  #     Y_synthetic = np.hstack((Y_synthetic, y_synthetic))

  # # Fit model-free model on new dataset
  X_new = np.vstack((np.vstack(env.X), X_synthetic))
  y_new = np.hstack((np.hstack(env.y), Y_synthetic))
  # infected_indices = np.where(convert_first_order_to_infection_status(X_new) == 1)

  # q0 = SKLogit2()
  # q0.fit(X_new, y_new, None, False, infected_indices, None)
  q0 = Ridge()
  q0.fit(X_new, y_new)

  # Define q-function
  def qfn(a):
    x = env.data_block_at_action(-1, a)
    # infected_indices = np.where(env.Y[-1, :] == 1)[0]
    # return q0.predict_proba(x, infected_indices, None)
    return q0.predict(x)

  a_ = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a_, {}

