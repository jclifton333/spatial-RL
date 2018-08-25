from src.estimation.q_functions.fqi import fqi, rollout_variance_estimate
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q
from src.estimation.model_based.sis.estimate_mb_q_fn import estimate_SIS_q_fn
from src.estimation.model_based.sis.fit import fit_transition_model
from src.estimation.q_functions.one_step import *

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

  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, weights, env)

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)[:, -1]

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a, q_model = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer']

    auto_regressor = AutoRegressor(classifier, regressor)

    q_model = fqi(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer)
    q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
    a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, q_model


def sis_model_based_policy(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, bootstrap, \
    rollout_depth, gamma, classifier, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['bootstrap'], \
    kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']

  auto_regressor = AutoRegressor(classifier, regressor)
  new_q_model = estimate_SIS_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth,
                                  treatment_budget, evaluation_budget, argmaxer, bootstrap)

  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=new_q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, new_q_model


def sis_model_based_one_step(**kwargs):
  env, bootstrap, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['bootstrap'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  eta = fit_transition_model(env, bootstrap=bootstrap)
  one_step_q = partial(sis_infection_probability, y=env.current_infected, s=env.current_state, eta=eta,
                       omega=0, L=env.L, adjacency_lists=env.adjacency_list)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  return a, None


def sis_one_step_mse_averaged(**kwargs):
  env = kwargs['env']

  alpha_mb, alpha_mf, q_mb, q_mf = one_step_sis_convex_combo(env)

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
  info = {'mb_bias': mb_bias, 'mb_var': mb_var, 'mf_var': mf_var, 'cov': mb_mf_cov, 'mf_bias': mf_bias}
  info.update({'alpha_mb': alpha_mb})
  return a, info


def sis_mse_averaged(**kwargs):
  classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer']

  X, X_raw = np.vstack(env.X), np.vstack(env.X_raw)
  y = np.hstack(y)

  # Fit one-step
  alpha_mb, alpha_mf, q_mb, q_mf = one_step_sis_convex_combo(env)
  infection_probabilities = alpha_mb*q_mb(X_raw) + alpha_mf*q_mf(X)
  regressor.fitRegressor(X, infection_probabilities, None, False)

  # Fit k-steps
  for k in range(1, rollout_depth + 1):
    # Estimate bias and variance of mb
    # Estimate variance of mf
    mf_var = rollout_variance_estimate(k, gamma, env, evaluation_budget, treatment_budget, regressor, argmaxer,
                                       infection_probabilities)

  return


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