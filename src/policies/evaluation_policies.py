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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.special import expit, logit
from scipy.stats import normaltest
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


def two_step_mb_myopic(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, bootstrap, gamma = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['bootstrap'], kwargs['gamma']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
    weights_1 = weights[env.L:]
  else:
    weights = None
    weights_1 = None

  # One step
  clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
  def qfn_at_block(block_index, a):
    return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  # ToDo: check that argsort is in descending order
  for t in range(1, env.T):
    # Get myopic action using true probs
    x_raw = env.X_raw[t]
    probs_t = env.infection_probability(np.zeros(env.L), x_raw[:, 2], x_raw[:, 0])
    a_myopic_t = np.zeros(env.L)
    a_myopic_t[np.argsort(probs_t)[-treatment_budget:]] = 1

    # Evaluate q0 at myopic action
    backup_at_t = qfn_at_block(t, a_myopic_t)
    backup.append(backup_at_t)

  # Fit backup-up q function
  # reg = regressor(n_estimators=100)
  reg = regressor()
  reg.fit(np.vstack(env.X[:-1]), np.hstack(backup), sample_weight=weights_1)

  return None, {'q_fn_params': reg.coef_}


def two_step_mb_constant_cutoff(**kwargs):
  CUTOFF = 0.5
  classifier, regressor, env, evaluation_budget, treatment_budget, bootstrap, gamma = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['bootstrap'], kwargs['gamma']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
    weights_1 = weights[env.L:]
    random_penalty_correction = np.sum(weights_1) / len(weights_1)
  else:
    weights = None
    weights_1 = None
    random_penalty_correction = 1.
    
  # One step
  # clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
  clf = LogisticRegression(C=1/random_penalty_correction, fit_intercept=False)
  clf.fit(np.vstack(env.X)[:, :8], np.hstack(env.y))
  def qfn_at_block(block_index, a):
    # return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)
    return clf.predict_proba(env.data_block_at_action(block_index, a)[:, :8])[:, -1]

  # Back up once
  backup = []

  for t in range(1, env.T):
    # Get myopic action using true probs
    x_raw = env.X_raw[t]
    probs_t = env.infection_probability(np.zeros(env.L), x_raw[:, 2], x_raw[:, 0])
    a_myopic_t = np.zeros(env.L)
    a_myopic_t[np.where(probs_t > CUTOFF)] = 1

    # Evaluate q0 at myopic action
    backup_at_t = qfn_at_block(t, a_myopic_t)
    backup.append(backup_at_t)

  # Fit backup-up q function
  # reg = regressor(n_estimators=100)
  # reg = regressor()

  # Adaptively choose ridge penalty based on eigenvals 
  X_stack = np.vstack(env.X[:-1])
  keep_ixs = [i for i in range(8)]
  X_stack = X_stack[:, keep_ixs]
  XpX = np.dot(X_stack.T, X_stack)
  eigs = np.linalg.eig(XpX)[0]
  # alpha_ = np.max(eigs)
  alpha_ = 1
  
  # Fit regression
  # reg = Ridge(alpha=alpha_*random_penalty_correction, fit_intercept=False)
  # ys = np.hstack(backup)
  reg = LogisticRegression(C=1/(alpha_*random_penalty_correction), fit_intercept=False)
  ys = np.hstack(env.y[:-1])
  reg.fit(X_stack, ys, sample_weight=weights_1)
  coefs = []
  for it in range(1000):
    weights = np.random.exponential(size=len(ys))
    reg = LogisticRegression(C=1/(alpha_*np.mean(weights)), fit_intercept=False)
    # reg = Ridge(alpha=alpha_*np.mean(weights), fit_intercept=False)
    reg.fit(X_stack, ys, sample_weight=weights)
    coefs.append(reg.coef_)

  # Count number of nonzero params
  X_nonzero = X_stack > 0
  X_nonzero_counts = X_nonzero.sum(axis=0)
  return None, {'q_fn_params': reg.coef_, 'nonzero_counts': X_nonzero_counts, 'eigs': eigs}

def two_step_mb_constant_cutoff_test(**kwargs):
  CUTOFF = 0.5
  N_REP = 100
  classifier, regressor, env, evaluation_budget, treatment_budget, bootstrap, gamma = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['bootstrap'], kwargs['gamma']

  coefs = [] 
  coefs_1 = []
  eigs = []
  eigs_1 = []
  for rep in range(N_REP):
    weights = np.random.exponential(size=len(env.X)*env.L)
    weights_1 = weights[env.L:]
    random_penalty_correction = np.sum(weights_1) / len(weights_1)
      
    # One step
    clf = LogisticRegression(C=1/(np.mean(weights)), fit_intercept=False)
    clf.fit(np.vstack(env.X)[:, :8], np.hstack(env.y))
    def qfn_at_block(block_index, a):
      return clf.predict_proba(env.data_block_at_action(block_index, a)[:, :8])[:, -1]

    # Back up once
    backup = []
    for t in range(1, env.T):
      # Get myopic action using true probs
      x_raw = env.X_raw[t]
      probs_t = env.infection_probability(np.zeros(env.L), x_raw[:, 2], x_raw[:, 0])
      a_myopic_t = np.zeros(env.L)
      a_myopic_t[np.where(probs_t > CUTOFF)] = 1

      # Evaluate q0 at myopic action
      backup_at_t = qfn_at_block(t, a_myopic_t)
      backup.append(backup_at_t)

    # Fit backup-up q function
    # reg = regressor(n_estimators=100)
    # reg = regressor()

    # Adaptively choose ridge penalty based on eigenvals 
    X_stack_1 = np.vstack(env.X[:-1])
    X_stack = np.vstack(env.X_raw[:-1])
    XpX = np.dot(X_stack.T, X_stack)
    XpX_1 = np.dot(X_stack_1.T, X_stack_1)
    alpha_ = 1
    
    # Fit regression with raw features
    reg = Ridge(alpha=alpha_*random_penalty_correction, fit_intercept=False)
    ys = np.hstack(backup)
    reg.fit(X_stack, ys, sample_weight=weights_1)
    coefs.append(reg.coef_)

    # Fit regression with full features
    reg_1 =Ridge(alpha=alpha_*random_penalty_correction, fit_intercept=False)
    reg_1.fit(X_stack_1, ys, sample_weight=weights_1)
    coefs_1.append(reg_1.coef_)

    # Collect eigenvalues
    eigs.append(np.linalg.eig(XpX / XpX.shape[0])[0])
    eigs_1.append(np.linalg.eig(XpX_1 / XpX_1.shape[0])[0])

  coefs = np.array(coefs)
  coefs_1 = np.array(coefs_1)
  eigs_var = np.var(eigs, axis=0)
  eigs_var_1 = np.var(eigs_1, axis=0)
  pdb.set_trace()
  # Count number of nonzero params
  X_nonzero = X_stack > 0
  X_nonzero_counts = X_nonzero.sum(axis=0)
  return None, {'q_fn_params': reg.coef_, 'nonzero_counts': X_nonzero_counts, 'eigs': eigs}



