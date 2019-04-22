import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
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




