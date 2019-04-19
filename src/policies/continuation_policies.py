"""
Policies of the form

finite-horizon mf qfunction + infinite horizon mb qfunction.
"""
import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
from src.policies.policy_search import policy_search, features_for_priority_score
import src.policies.q_function_policies as qfn_policies
import numpy as np


def one_step_continuation(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  # # ToDo: Using random actions for diagnostic purposes!
  # a = np.concatenate((np.zeros(env.L - treatment_budget), np.ones(treatment_budget)))
  # a = np.random.permutation(a)

  return a, loss_dict