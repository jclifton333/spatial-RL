import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.q_functions.model_fitters import SKLogit2
from src.estimation.q_functions.one_step import *
from src.environments.sis import convert_first_order_to_infection_status
import numpy as np


def sis_one_step_dyna(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, gamma = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma']
  q_mb_one_step, _ = fit_one_step_sis_mb_q(env)

  MAX_NUM_NONZERO = np.min((env.max_num_neighbors, 8))
  QUOTA = int(np.sqrt(env.L * env.T))

  bools = (0, 1)
  raw_feature_combos = [[i, j, k] for i in bools for j in bools for k in bools]

  # Build dictionary of feature indicator counts
  # ToDo: this shouldn't be done each time the policy is called...
  unique_feature_indicators = {}
  for n in range(MAX_NUM_NONZERO):
    truth_vals = [1 for i in range(n)] + [0 for i in range(8 - n)]
    for permutation_ in all_permutations(truth_vals):
      for raw_feature_combo in raw_feature_combos:
        feature_combo = tuple(raw_feature_combo + permutation_)
        unique_feature_indicators[feature_combo] = {'count': 0, 'list': []}

  mb_phats = [q_mb_one_step(x) for x in env.X]

  # Count number of feature indicators
  for t, X_ in enumerate(env.X):
    for l, x in enumerate(X_):
      x_indicator = x > 0
      unique_feature_indicators[tuple(x_indicator)]['count'] += 1  
      unique_feature_indicators[tuple(x_indicator)]['list'].append((x, t, l))

  # Add current state, infection info evaluated at treatment and no-treatment
  X_current_trt = env.data_block_at_action(-1, np.zeros(env.L))
  X_current_no_trt = env.data_block_at_action(-1, np.ones(env.L))
  for l in range(env.L):
    x_current_trt_indicator = X_current_trt[l] > 0
    x_current_no_trt_indicator = X_current_no_trt[l] > 0
    unique_feature_indicators[tuple(x_current_trt_indicator)]['count'] += 1
    unique_feature_indicators[tuple(x_current_trt_indicator)]['list'].append((x_current_trt_indicator, env.T, l))
    unique_feature_indicators[tuple(x_current_no_trt_indicator)]['count'] += 1
    unique_feature_indicators[tuple(x_current_no_trt_indicator)]['list'].append((x_current_no_trt_indicator, env.T, l))

  # Supplement features that fall short of quota
  X_synthetic = np.zeros((0, env.X[0].shape[1]))
  Y_synthetic = np.zeros(0)
  for feature_info in unique_feature_indicators.values():
    count = feature_info['count']
    if 0 < count < QUOTA:  # ToDo: what if count=0?
      num_fake_data = QUOTA - count

      # Sample with replacement up to desired number
      feature_list = feature_info['list']
      synthetic = np.random.choice(feature_list, num_fake_data, replace=T)
      x_synthetic = [o_[0] for o_ in synthetic]
      y_synthetic = [mb_phats[o_[1]][o_[2]] for o_ in synthetic]

      # Add to dataset
      X_synthetic = np.vstack((X_synthetic, x_synthetic))
      Y_synthetic = np.hstack((Y_synthetic, y_synthetic))

  # Fit model-free model on new dataset
  X_new = np.vstack((np.vstack(env.X), X_synthetic))
  y_new = np.hstack((env.y, Y_synthetic))
  infected_indices = [ix for ix in range(X_new.shape[0]) if convert_first_order_to_infection_status(X_new[ix])]

  q0 = SKLogit2()
  q0.fit(X_new, y_new, infected_indices, None)

  # Define q-function
  def qfn(a):
    x = env.data_block_at_action(-1, a)
    infected_indices = np.where(env.Y[-1, :] == 1)
    return q0.predict_proba(x, infected_indices, None)

  a_ = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a_, {}

