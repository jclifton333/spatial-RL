"""
Functions for generating bootstrap Bellman error distributions.
"""

import pdb
from src.estimation.model_based.sis.estimate_mb_q_fn import estimate_SIS_q_fn
from src.estimation.stacking.compute_sample_bellman_error import compute_sample_squared_bellman_error
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.rollout import rollout
from keras import backend as K


def bootstrap_SIS_mb_qfn(env, classifier, regressor, rollout_depth, gamma, planning_depth,
                         treatment_budget, evaluation_budget, argmaxer, num_bootstrap_samples):
  be_list = []
  for rep in range(num_bootstrap_samples):
    auto_regressor = AutoRegressor(classifier, regressor)
    q_fn = estimate_SIS_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth,
                             treatment_budget, evaluation_budget, argmaxer, bootstrap=True)
    bootstrap_be = compute_sample_squared_bellman_error(q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                        argmaxer)
    be_list.append(float(bootstrap_be))
    K.clear_session()
  return be_list


def bootstrap_rollout_qfn(env, classifier, regressor, rollout_depth, gamma, treatment_budget,
                          evaluation_budget, argmaxer, num_bootstrap_samples):
  be_list = []
  for rep in range(num_bootstrap_samples):
    print(rep)
    auto_regressor = AutoRegressor(classifier, regressor)
    q_fn = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                   bootstrap=True)
    bootstrap_be = compute_sample_squared_bellman_error(q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                        argmaxer)
    be_list.append(float(bootstrap_be))
    K.clear_session()
  return be_list
