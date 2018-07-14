"""
Functions for generating bootstrap Bellman error distributions.
"""

from src.estimation.model_based.SIS.estimate_mb_q_fn import estimate_SIS_q_fn
from src.estimation.stacking.compute_sample_bellman_error import compute_sample_squared_bellman_error
from src.estimation.q_functions.rollout import rollout


def bootstrap_SIS_mb_qfn(env, classifier, regressor, rollout_depth, gamma, planning_depth, q_model,
                         treatment_budget, evaluation_budget, argmaxer, num_bootstrap_samples):
  BE_list = []
  for rep in range(num_bootstrap_samples):
    q_fn = estimate_SIS_q_fn(env, classifier, regressor, rollout_depth, gamma, planning_depth, q_model,
                             treatment_budget, evaluation_budget, argmaxer, train_ixs=None, bootstrap=True)
    bootstrap_BE = compute_sample_squared_bellman_error(q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                        argmaxer)
    BE_list.append(bootstrap_BE)
  return BE_list


def bootstrap_rollout_qfn(env, auto_regressor, rollout_depth, gamma, treatment_budget,
                          evaluation_budget, argmaxer, num_bootstrap_samples):
  BE_list = []
  for rep in range(num_bootstrap_samples):
    q_fn = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                   bootstrap=True)
    bootstrap_BE = compute_sample_squared_bellman_error(q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                        argmaxer)
    BE_list.append(bootstrap_BE)
  return BE_list
