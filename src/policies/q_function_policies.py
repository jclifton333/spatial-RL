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

import numpy as np
import keras.backend as K
from functools import partial


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights)

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def two_step_mb(**kwargs):
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  if env.__class__.__name__ == 'sis':
    qfn_at_block_, predict_proba_kwargs = fit_one_step_sis_mb_q(env, bootstrap_weights=weights)
  elif env.__class__.__name__ == 'Ebola':
    q_fn_at_block_, _ = fit_one_step_ebola_mb_q(env)

  def q_fn(raw_data_block, a):
    raw_data_block_at_action = np.column_stack((raw_data_block[:, 0], a, raw_data_block[:, 2]))
    return qfn_at_block_(raw_data_block_at_action)

  # Back up once
  backup = []
  for t in range(env.T-1):
    q_fn_at_block_t = lambda a: q_fn_at_block_(np.column_stack((env.X_raw[t][:, 0], a, env.X_raw[t][:, 2])))
    a_max = argmaxer(q_fn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = q_fn_at_block_t(a_max)
    backup_at_t = env.y[t] + q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  reg = regressor()
  reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

  def backedup_up_qfn(a):
    return reg.predict(env.data_block_at_action(-1, a))

  a = argmaxer(backedup_up_qfn, evaluation_budget, treatment_budget, env)
  return a, None


def two_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights)
  def qfn_at_block(block_index, a):
    return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(env.T-1):
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  reg = regressor()
  reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

  def qfn(a):
    X_ = env.data_block_at_action(-1, a)
    X2 = env.data_block_at_action(-1, a, neighbor_order=2)
    return clf.predict(X_) + reg.predict(X2)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def two_step_stacked(**kwargs):
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

    # Fit rewards
    if env.__class__.__name__ == 'SIS':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])

    elif env.__class__.__name__ == 'Ebola':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])

    def q_mb_at_block(t, a):
      return q_mb_fold(env.data_block_at_action(t, a, raw=True))

    def q_mf_at_block(t, a):
      X_raw_t = env.X_raw[t]
      infected_ixs = np.where(X_raw_t[:, -1] == 1)
      not_infected_ixs = np.where(X_raw_t[:, -1] == 0)
      return q_mf_fold(env.data_block_at_action(t, a), infected_ixs, not_infected_ixs)

    # Back up once
    backup_mb = []
    backup_mf = []
    for t in range(env.T-1):
      train_ixs = train_test_split[0][t]
      q_mf_block_t = lambda a: q_mf_fold(t, a)
      q_mb_block_t = lambda a: q_mb_fold(t, a)
      a_max_mf = argmaxer(q_mf_block_t, evaluation_budget, treatment_budget, env)
      a_max_mb = argmaxer(q_mb_block_t, evaluation_budget, treatment_budget, env)
      q_max_mf = q_mf_at_block(a_max_mf)[train_ixs]
      q_max_mb = q_mb_at_block(a_max_mb)[train_ixs]

      backup_at_t_mf = env.y[t][train_ixs] + q_max_mf
      backup_at_t_mb = q_mb_fold(env.X_raw[t])[train_ixs] + q_max_mb
      backup_mf.append(backup_at_t_mf)
      backup_mb.append(backup_at_t_mb)

    # Fit backed-up q fns
    reg_mf = regressor()
    reg_mb = regressor()
    X_train = np.vstack([x[train_test_split[0][t]] for (t, x) in enumerate(env.X)])
    reg_mf.fit(X_train, backup_mf)
    reg_mb.fit(X_train, backup_mb)

    # Get backup values on test set
    X_test = np.vstack([x[train_test_split[1][t]] for (t, x) in enumerate(env.X)])
    reg_mf.predict(X_test)
    reg_mb.predict(X_test)

    y = np.array([y_[train_test_split[1][t]] for (t, y_) in enumerate(env.y)])
    for t, (x_raw, x) in enumerate(zip(env.X_raw[:-1], env.X[:-1])):
      test_ixs = train_test_split[1][t]
      qhat_mb = np.append(qhat_mb, reg_mb.predict())

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


def two_step_higher_order(**kwargs):
  """
  Use second-order neighbor features rather than first order (as in two_step).
  :param kwargs:
  :return:
  """
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights)
  def qfn_at_block(block_index, a):
    return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(env.T-1):
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = env.y[t] + q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  reg = regressor()
  reg.fit(np.vstack(env.X_2[:-1]), np.hstack(backup))

  def qfn(a):
    return reg.predict(env.data_block_at_action(-1, a, neighbor_order=2))

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


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


def gravity_model_based_one_step(**kwargs):
  env, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  if env.__class__.__name__ == 'Ebola':
    eta = fit_ebola_transition_model(env)
  elif env.__class__.__name__ == 'ContinuousGrav':
    eta = fit_continuous_grav_transition_model(env)
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


def continuous_model_based_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  if env.__class__.__name__ == 'Ebola':
    eta = fit_ebola_transition_model(env)
  elif env.__class__.__name__ == 'ContinuousGrav':
    eta = fit_continuous_grav_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = np.zeros(env.L)
  probs = one_step_q(a)

  # Get priority score
  priorities_for_infected_locations = np.zeros(env.L)
  for l in np.where(env.current_infected == 1)[0]:
    for lprime in env.adjacency_list[l]:
      priorities_for_infected_locations[l] += \
        (1 - env.current_infected[lprime]) * probs[lprime] / env.distance_matrix[l, lprime]
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
