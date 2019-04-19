"""
Policies of the form

finite-horizon mf qfunction + infinite horizon mb qfunction.
"""
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
import src.policies.policy_search as ps
import src.policies.q_function_policies as qfn_policies
import numpy as np


def sis_policy_search_continuation(policy_parameter_, number_of_steps_ahead, env, x_raw_current, remaining_time_horizon,
                                   treatment_budget, infection_probs_predictor, infection_probs_kwargs,
                                   transmission_probs_predictor, transmission_probs_kwargs, data_depth):
  """
  Estimate returns under policy indexed by policy_parameter_, starting number_of_steps_ahead.

  :param policy_parameter_:
  :param number_of_steps_ahead:
  :param env:
  :param x_raw_current:
  :param remaining_time_horizon:
  :return:
  """
  k = 1
  s, a, y = x_raw_current[:, 0], x_raw_current[:, 1], x_raw_current[:, 2]
  score = ps.roll_out_candidate_policy(remaining_time_horizon, s, a, y, beta, policy_parameter_, treatment_budget, k,
                                       env, infection_probs_predictor,
                                       infection_probs_kwargs, transmission_probs_predictor, transmission_probs_kwargs,
                                       data_depth, number_of_steps_ahead=number_of_steps_ahead, monte_carlo_reps=10,
                                       gamma=0.9)


def sis_one_step_continuation(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  # Get optimal policy
  policy_parameter_ = ps.policy_parameter_wrapper(**kwargs)

  # Fit one-step model free qfn
  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None
  clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, loss_dict