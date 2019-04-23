import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.q_functions.fqi import fqi
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q, q_max_all_states
from src.estimation.model_based.sis.estimate_sis_q_fn import estimate_sis_q_fn
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
from src.estimation.model_based.Gravity.estimate_continuous_parameters import fit_continuous_grav_transition_model
from src.estimation.q_functions.model_fitters import SKLogit2
import src.estimation.q_functions.mse_optimal_combination as mse_combo
from src.estimation.q_functions.one_step import *
from src.utils.misc import random_argsort
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.special import expit, logit
import numpy as np
import keras.backend as K
from functools import partial


def sis_one_step_dyna(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, gamma, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma'], kwargs['regressor']
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

  # Supplement features that fall short of quota
  X_synthetic = np.zeros((0, env.X[0].shape[1]))
  Y_synthetic = np.zeros(0)
  for feature_info in unique_feature_indicators.values():
    count = feature_info['count']  # ToDo: what if count=0?
    if count < QUOTA:
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
  infected_indices = None
  q0 = SKLogit2()
  q0.fit(X_new, y_new, infected_indices, None)

  # Define q-function
  def qfn(a):
    x = env.data_block_at_action(-1, a)
    infected_indices = np.where(env.Y[-1, :] == 1)
    return q0.predict_proba(x, infected_indices, None)

  a_ = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a_, {}
