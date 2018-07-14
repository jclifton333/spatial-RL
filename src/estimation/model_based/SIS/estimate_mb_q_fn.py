from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.rollout import rollout
from .fit import fit_transition_model
from .simulate import simulate_from_SIS


def estimate_SIS_q_fn(env, classifier, regressor, rollout_depth, gamma, planning_depth, q_model, treatment_budget,
                      evaluation_budget, argmaxer, train_ixs, bootstrap):

  auto_regressor = AutoRegressor(classifier, regressor)

  # Need to fit q_model if it hasn't been already
  if q_model is None:
    q_model = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer)

  # Estimate MDP and generate data using policy = argmax q_model
  eta = fit_transition_model(env, bootstrap=bootstrap, ixs=train_ixs)
  simulation_env = simulate_from_SIS(env, eta, planning_depth, q_model, argmaxer, evaluation_budget,
                                     treatment_budget)

  # Estimate optimal q-function from simulated data
  q_model = rollout(rollout_depth, gamma, simulation_env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                    ixs=train_ixs)

  return q_model
