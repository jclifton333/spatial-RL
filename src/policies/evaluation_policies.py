"""
Policies which are to be evaluated using Q-learning, but not used for decision-making itself.
"""
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
# import keras.backend as K
from functools import partial


def two_step_random(**kwargs):
  NUM_RANDOM_DRAWS = 100
  classifier, regressor, env, evaluation_budget, treatment_budget, bootstrap, gamma = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['bootstrap'], kwargs['gamma']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
  def qfn_at_block(block_index, a):
    return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # ToDo: Comment in after debugging
  # Back up once
  backup = []
  for t in range(1, env.T):
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_dummy = np.hstack((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
    q_randos = []
    for random_draw in range(NUM_RANDOM_DRAWS):
      a_rando = np.random.permutation(a_dummy)
      q_rando = qfn_at_block_t(a_rando)
      q_randos.append(q_rando)
    backup_at_t = np.array(q_randos).mean(axis=0)
    backup.append(backup_at_t)

  # Fit backup-up q function
  # reg = regressor(n_estimators=100)
  reg = regressor()
  reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

  def qfn(a):
    infections = env.Y[-1, :]
    infected_indices = np.where(infections == 1)[0]
    not_infected_indices = np.where(infections == 0)[0]
    X_ = env.data_block_at_action(-1, a)
    # ToDo: Comment back in after debugging
    # X = env.data_block_at_action(-1, a, neighbor_order=2)
    return clf.predict_proba(X_, infected_indices, not_infected_indices) + gamma * reg.predict(X_)
    # return clf.predict_proba(X_, infected_indices, not_infected_indices)

  return None, {'q_fn_params': reg.coef_}


def two_step_mb_eval(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, gamma, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma'], kwargs['regressor']

  # Compute backup
  q_mb_one_step, _ = fit_one_step_sis_mb_q(env)
  mb_backup = mse_combo.sis_mb_backup(env, gamma, q_mb_one_step, q_mb_one_step, argmaxer, evaluation_budget,
                                      treatment_budget)

  # Fit q-function
  X_2 = np.vstack(env.X_2[:-1])
  reg = regressor()
  reg.fit(X_2, mb_backup)

  return None, {'q_fn_params': reg.coef_}

