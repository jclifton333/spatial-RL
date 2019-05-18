import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
import src.environments.sis_infection_probs as sis_infection_probs
from src.estimation.q_functions.one_step import fit_one_step_predictor
from src.policies.policy_search import policy_search, features_for_priority_score
import src.policies.q_function_policies as qfn_policies
import numpy as np


def sis_aic_one_step(**kwargs):
  mf_classifier, env = kwargs['classifier'], kwargs['env']

  # Get aic of model-free and model-based estimators
  infected_locations = np.where(np.vstack(env.X_raw)[:, :-1] == 1)
  clf = mf_classifier()
  clf.fit(np.vstack(env.X), np.hstack(env.y), None, False, infected_locations, None)
  mf_aic = clf.aic
  beta_mean, mb_aic = fit_infection_prob_model(env, None)

  if mf_aic < mb_aic:
    return qfn_policies.one_step_policy(**kwargs)
  else:
    return qfn_policies.sis_model_based_one_step(**kwargs)


def sis_local_aic_one_step(**kwargs):
  mf_classifier, env, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['classifier'], kwargs['env'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']

  # Get weighted likelihood of each observation under sis model and model-free classifier
  beta_, _ = fit_infection_prob_model(env, None)
  mb_log_likelihoods = np.array([])
  # ToDo: check indexing to make sure this is actually the current x
  x_current = env.X[-1]
  p_mf = x_current.shape[1]
  weights = np.zeros((env.L, 0))
  for x_raw, x, y_next in zip(env.X_raw, env.X, env.y):
    # Unweighted likelihoods
    s, a, y = x_raw[:, 0], x_raw[:, 1], x_raw[:, 2]
    mb_probs = sis_infection_probs.sis_infection_probability(a, y, beta_, env.L, env.adjacency_list,
                                                             **{'omega': 0.0, 's': s})
    mb_probs_clipped = np.min((np.max((0.001, mb_probs)), 0.999))  # For stability
    mb_log_likelihoods_at_x_raw = y * np.log(mb_probs_clipped) + (1 - y) * np.log(1 - mb_probs_clipped)
    mb_log_likelihoods = np.append((mb_log_likelihoods, mb_log_likelihoods_at_x_raw))

    # Weight by distances to each current x
    weights_at_x = np.array([[1 - np.sum(x_current[l] == x[lprime])/p for lprime in range(env.L)] for l in range(env.L)])
    weights = np.column_stack((weights, weights_at_x))

  # MF log likelihoods
  infected_locations = np.where(np.vstack(env.X_raw)[:, :-1] == 1)
  clf = mf_classifier()
  clf.fit(np.vstack(env.X), np.vstack(env.y), None, False, infected_locations, None)
  mf_log_likelihoods = clf.log_likelihood_elements

  # Weight likelihoods by distances from current observation
  summed_weights = weights.sum(axis=1)
  local_mb_log_likelihoods = np.dot(weights, mb_log_likelihoods)
  local_mb_log_likelihoods /= summed_weights
  local_mf_log_likelihoods = np.dot(weights, mf_log_likelihoods)
  local_mf_log_likelihoods /= summed_weights

  # Get local AIC of each model at each current location
  p_mb = len(beta_)
  local_mf_AIC = -local_mf_log_likelihoods + p_mf
  local_mb_AIC = -local_mb_log_likelihoods + p_beta
  use_mb_probability = local_mb_AIC < local_mf_AIC

  # Define q-function and get action
  clf, predict_proba_kwargs, _ = fit_one_step_predictor(mf_classifier, env, None)
  y_current = env.Y[-1, :]
  infected_locations = np.where(y_current)

  def qfn(a):
    x_at_a = env.data_block_at_action(-1, a, neighbor_order=1)

    phat_mf = clf.predict_proba(x_at_a, infected_locations, None)
    phat_mb = sis_infection_probs.sis_infection_probability(a, y_current, beta_, env.L, env.adjacency_list,
                                                            **{'omega': 0.0, 's': np.zeros(env.L)})
    phat = phat_mb * use_mb_probability + phat_mf * (1 - use_mb_probability)
    return phat

  a_ = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a_


def ebola_aic_one_step(**kwargs):
  mf_classifier, env = kwargs['classifier'], kwargs['env']

  # Get aic of model-free and model-based estimators
  infected_locations = np.where(np.vstack(env.X_raw)[:, :-1] == 1)
  clf = mf_classifier()
  clf.fit(np.vstack(env.X), np.hstack(env.y), None, False, infected_locations, None)
  mf_aic = clf.aic
  beta_mean, mb_aic = fit_ebola_transition_model(env, None)

  if mf_aic < mb_aic:
    return qfn_policies.one_step_policy(**kwargs)
  else:
    return qfn_policies.gravity_model_based_one_step(**kwargs)


def sis_aic_two_step(**kwargs):
  """
  Use AIC to select model-free or model-based probability estimator.  If model-based is selected, use policy search.
  If model-free is selected, use two-step.

  :param kwargs:
  :return:
  """
  mf_classifier, env = kwargs['classifier'], kwargs['env']

  # Get aic of model-free and model-based estimators
  infected_locations = np.where(np.vstack(env.X_raw)[:, :-1] == 1)
  clf = mf_classifier()
  clf.fit(np.vstack(env.X), np.hstack(env.y), None, False, infected_locations, None)
  mf_aic = clf.aic
  beta_mean, mb_aic = fit_infection_prob_model(env, None)

  if mf_aic < mb_aic:  # Model-free policy
    return qfn_policies.two_step(**kwargs)
  else:  # Model-based policy search
    treatment_budget, remaining_time_horizon, initial_policy_parameter = \
      kwargs['treatment_budget'], kwargs['planning_depth'], kwargs['initial_policy_parameter']
    beta_mean, _ = fit_infection_prob_model(env, None)
    beta_cov = env.mb_covariance(beta_mean)

    def gen_model_posterior():
      beta_tilde = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)
      return beta_tilde

    # Settings
    if initial_policy_parameter is None:
      initial_policy_parameter = np.ones(3) * 0.5
    initial_alpha = initial_zeta = None
    # remaining_time_horizon = T - env.T

    # ToDo: These were tuned using bayes optimization on 10 mc replicates from posterior obtained after 15 steps of random
    # ToDo: policy; may be improved...
    rho = 3.20
    tau = 0.76
    a, policy_parameter = policy_search(env, remaining_time_horizon, gen_model_posterior, initial_policy_parameter,
                                        initial_alpha, initial_zeta, treatment_budget, rho, tau, tol=1e-3,
                                        maxiter=100, feature_function=features_for_priority_score, k=1,
                                        method='bayes_opt')
    return a, {'initial_policy_parameter': policy_parameter}




