from src.estimation.q_functions.fqi import fqi
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q, q_max_all_states
from src.estimation.model_based.sis.estimate_sis_q_fn import estimate_sis_q_fn
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.model_based.Ebola.estimate_ebola_parameters import fit_ebola_transition_model
from src.estimation.q_functions.model_fitters import SKLogit2
import src.estimation.q_functions.mse_optimal_combination as mse_combo
from src.estimation.q_functions.one_step import *
from src.utils.misc import random_argsort

import numpy as np
import keras.backend as K
from functools import partial


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, truncate = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['truncate']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights, truncate)

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a, q_model = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer']

    q_model = fqi(rollout_depth, gamma, env, evaluation_budget, treatment_budget, classifier, regressor, argmaxer)
    q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
    a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, q_model


def sis_mb_fqi(**kwargs):
  """
  Currently only two-step fqi!

  :param kwargs:
  :return:
  """
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

  # Get optimal action
  def qfn(action):
    return q(action, -1, env, reg.predict, neighbor_order=2)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)

  return a, None


def sis_model_based_policy(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, bootstrap, \
    rollout_depth, gamma, classifier, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['bootstrap'], \
    kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']

  auto_regressor = AutoRegressor(classifier, regressor)
  new_q_model = estimate_sis_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth,
                                  treatment_budget, evaluation_budget, argmaxer, bootstrap)

  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=new_q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, new_q_model


def ebola_model_based_one_step(**kwargs):
  env, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  eta = fit_ebola_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  phat = one_step_q(a)
  ptrue = env.next_infected_probabilities(a)
  diff = np.abs(phat - ptrue)
  worst_ix = np.argmax(diff)
  print('eta: {}\ntrue eta: {}'.format(eta, env.ETA))
  # print('max loss: {} mean loss: {} worst ix: {}'.format(np.max(diff), np.mean(diff), worst_ix))
  return a, None


def sis_model_based_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  eta = fit_sis_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = np.zeros(env.L)
  probs = one_step_q(a)

  treat_ixs = random_argsort(probs, treatment_budget)
  a[treat_ixs] = 1
  return a, None


def ebola_model_based_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  eta = fit_ebola_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = np.zeros(env.L)
  probs = one_step_q(a)

  # Get priority score
  priorities_for_infected_locations = np.zeros(env.L)
  for l in np.where(env.current_infected == 1)[0]:
    for lprime in env.adjacency_list[l]:
      priorities_for_infected_locations[l] += \
        (1 - env.current_infected[lprime]) * probs[lprime] / env.DISTANCE_MATRIX[l, lprime]
  priorities_for_infected_locations /= np.sum(priorities_for_infected_locations)
  priority = probs

  # Treat greedily acc to priority
  priority[np.where(env.current_infected == 1)] = \
    priorities_for_infected_locations[np.where(env.current_infected == 1)]
  treat_ixs = random_argsort(-priority, treatment_budget)
  a[treat_ixs] = 1

  return a, None


def sis_model_based_one_step(**kwargs):
  env, bootstrap, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['bootstrap'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  eta = fit_sis_transition_model(env)
  print('eta hat: {} eta true: {}'.format(eta, env.ETA))
  one_step_q = partial(sis_infection_probability, y=env.current_infected, s=env.current_state, eta=eta,
                       omega=0, L=env.L, adjacency_lists=env.adjacency_list)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  return a, None


def one_step_mse_averaged(**kwargs):
  env = kwargs['env']

  res = mse_combo.one_step_convex_combo(env)

  alpha_mb, alpha_mf, q_mb, q_mf = res['alpha_mb'], res['alpha_mf'], res['q_mb'], res['q_mf']

  # Get modified q_function
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    return alpha_mb*q_mb(raw_data_block) + alpha_mf*q_mf(data_block, infected_indices[0], not_infected_indices[0])

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  # info = {'mb_bias': mb_bias, 'mb_var': mb_var, 'mf_var': mf_var, 'cov': mb_mf_cov, 'mf_bias': mf_bias}
  info = {}
  info.update({'alpha_mb': alpha_mb})
  return a, info


def one_step_stacked(**kwargs):
  N_SPLITS = 10
  TRAIN_PROPORTION = 0.8

  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  # Train-test splits
  train_test_splits = []  # List of tuples (training_ixs, test_ixs), where each of these is a list of lists of indices
  for fold in range(N_SPLITS):
    training_ixs_for_fold = []
    test_ixs_for_fold = []
    for t in range(env.T):
      train_test_mask = np.random.binomial(1, TRAIN_PROPORTION, size=env.L)
      training_ixs_for_fold.append(np.where(train_test_mask == 1)[0])
      test_ixs_for_fold.append(np.where(train_test_mask == 0)[0])
    train_test_splits.append((training_ixs_for_fold, test_ixs_for_fold))

  # Fit models on training splits
  yhat_mb = np.zeros(0)
  yhat_mf = np.zeros(0)
  y = np.zeros(0)
  for fold in range(N_SPLITS):
    train_test_split = train_test_splits[fold]
    if env.__class__.__name__ == 'SIS':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])

    elif env.__class__.__name__ == 'Ebola':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])
    for t, (x_raw, x) in enumerate(zip(env.X_raw[:-1], env.X[:-1])):
      test_ixs = train_test_split[1][t]
      yhat_mb = np.append(yhat_mb, q_mb_fold(x_raw)[test_ixs])
      yhat_mf = np.append(yhat_mf, q_mf_fold(x[test_ixs, :], np.where(x_raw[test_ixs, -1] == 1),
                                             np.where(x_raw[test_ixs, -1] == 0)))
      y = np.append(y, env.y[t][test_ixs])

  # Get optimal combination weight
  alpha_mb = np.sum(np.multiply(y - yhat_mf, yhat_mb - yhat_mf)) / np.linalg.norm(yhat_mb - yhat_mf)**2
  alpha_mb = np.min((1.0, np.max((0.0, alpha_mb))))

  # Stack q functions
  if env.__class__.__name__ == 'SIS':
    q_mb, q_mf, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)
  elif env.__class__.__name__ == 'Ebola':
    q_mb, q_mf, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2)

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    return alpha_mb * q_mb(raw_data_block) + \
           (1 - alpha_mb) * q_mf(data_block, infected_indices[0], not_infected_indices[0])

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  info = {}
  return a, info


def sis_one_step_equal_averaged(**kwargs):
  env = kwargs['env']
  
  if env.__class__.__name__ == 'SIS':
    q_mb, q_mf, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)
  elif env.__class__.__name__ == 'Ebola':
    q_mb, q_mf, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2)

  # Get modified q_function
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    return 0.5 * q_mb(raw_data_block) + 0.5 * q_mf(data_block, infected_indices[0], not_infected_indices[0])

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  info = {}
  return a, info


def sis_two_step_mse_averaged(**kwargs):
  regressor, env, evaluation_budget, gamma, treatment_budget, argmaxer = \
      kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['gamma'], kwargs['treatment_budget'], \
      kwargs['argmaxer']

  # Get mse-averaged backup
  averaged_backup, info = mse_combo.two_step_sis_convex_combo(env, gamma, argmaxer, evaluation_budget, treatment_budget)

  # Fit q-function to backup
  X_2 = np.vstack(env.X_2[:-1])
  reg = regressor()
  reg.fit(X_2, averaged_backup)

  def qfn(action):
    return q(action, -1, env, reg.predict, neighbor_order=2)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, info


# def dummy_stacked_q_policy(**kwargs):
#   """
#   This is for testing stacking.
#   """
#   gamma, env, evaluation_budget, treatment_budget, argmaxer = kwargs['gamma'], kwargs['env'], \
#     kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer']
#
#   # Generate dummy q functions
#   B = 2
#   q1_list = []
#   q2_list = []
#   bootstrap_weights_list = []
#   for b in range(B):
#     q1_b = lambda x: np.random.random(x.shape[0])
#     q2_b = lambda x: np.random.random(x.shape[0])
#     bs_weights_b = np.random.exponential(size=(env.T, env.L))
#     q1_list.append(q1_b)
#     q2_list.append(q2_b)
#     bootstrap_weights_list.append(bs_weights_b)
#
#   bootstrap_weight_correction_arr = compute_bootstrap_weight_correction(bootstrap_weights_list)
#   theta = ggq(q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, argmaxer,
#               bootstrap_weight_correction_arr=bootstrap_weight_correction_arr)
#   a = np.random.permutation(np.append(np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
#   new_q_model = lambda x: theta[0]*q_0(x) + theta[1]*q_1(x)
#   return a, new_q_model


# def sis_one_step_be_averaged_policy(**kwargs):
#   """
#   Average one step MB and MF policies using bootstrap distributions of bellman errors.
#   :param kwargs:
#   :return:
#   """
#   env, classifier, argmaxer, evaluation_budget, treatment_budget, gamma = \
#     kwargs['env'], kwargs['classifier'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
#     kwargs['gamma']
#   B = 30
#
#   # Bootstrap functions to get P( BE(MB) > BE(MF)), which we use to average the q-functions
#   bootstrapped_q_lists = bootstrap_one_step_q_functions(env, classifier, B)
#   q_mb_list, q_mf_list = bootstrapped_q_lists['q_mb_list'], bootstrapped_q_lists['q_mf_list']
#   q_mb_be = np.array([bellman_error(env, q_mb, evaluation_budget, treatment_budget, argmaxer, gamma,
#                                     use_raw_features=True) for q_mb in q_mb_list])
#   q_mf_be = np.array([bellman_error(env, q_mf, evaluation_budget, treatment_budget, argmaxer, gamma,
#                                     use_raw_features=False) for q_mf in q_mf_list])
#   mb_weight = np.mean(q_mb_be < q_mf_be)
#   mf_weight = 1 - mb_weight
#
#   # Fit non-bootstrapped q functions and combine for final policy
#   q_mb, q_mf = fit_one_step_mf_and_mb_qs(env, classifier)
#
#   # Compare accuracy
#
#   def averaged_q_model(data_block):
#     return mb_weight*q_mb(data_block) + mf_weight*q_mf(data_block)
#
#   q_stacked = partial(q, data_block_ix=-1, env=env, predictive_model=averaged_q_model)
#   a = argmaxer(q_stacked, evaluation_budget, treatment_budget, env)
#   return a, {'theta': [mb_weight, mf_weight]}
#
#
# def sis_one_step_stacked_q_policy(**kwargs):
#   env, classifier, gamma, evaluation_budget, treatment_budget, argmaxer = \
#     kwargs['env'], kwargs['classifier'], kwargs['gamma'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
#     kwargs['argmaxer']
#
#   # Get bootstrapped qs
#   B = 1  # Fix number of bootstrap replicates for now
#   bootstrap_q_lists = bootstrap_one_step_q_functions(env, classifier, B)
#
#   # Get stacking parameter
#   theta = stack(bootstrap_q_lists['q_mb_list'], bootstrap_q_lists['q_mf_list'], gamma, env, evaluation_budget,
#                 treatment_budget, argmaxer, bootstrap_weight_list=bootstrap_q_lists['bootstrap_weight_list'])
#
#   # Fit non-bootstrapped q functions and combine for final policy
#   q_mb, q_mf = fit_one_step_mf_and_mb_qs(env, classifier)
#
#   def stacked_q_model(data_block):
#     return theta[0]*q_mb(data_block) + theta[1]*q_mf(data_block)
#
#   q_stacked = partial(q, data_block_ix=-1, env=env, predictive_model=stacked_q_model)
#   a = argmaxer(q_stacked, evaluation_budget, treatment_budget, env)
#   return a, {'theta': theta}
#
#
# def sis_stacked_q_policy(**kwargs):
#   env = kwargs['env']
#   train_ixs, test_ixs = env.train_test_split()
#   kwargs['train_ixs'] = train_ixs
#   _, q_mf = rollout_policy(**kwargs)
#   _, q_mb = SIS_model_based_policy(**kwargs)
#   q_list = [q_mf, q_mb]
#   gamma, evaluation_budget, treatment_budget, argmaxer = kwargs['gamma'], kwargs['evaluation_budget'], \
#                                                          kwargs['treatment_budget'], kwargs['argmaxer']
#   ixs = [0, 1, 2]
#   ixs_list = [ixs for _ in range(env.T)]
#   theta = ggq(q_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, ixs_list)
#   q_model = lambda x: theta[0]*q_mf(x) + theta[1]*q_mb(x)
#   q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
#   a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
#    return a, q_hat
