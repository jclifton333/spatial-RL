import pdb
from src.estimation.q_functions.rollout import rollout
from .fit import fit_transition_model
from .simulate import simulate_from_SIS


def estimate_SIS_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth, q_model, treatment_budget,
                      evaluation_budget, argmaxer, train_ixs, bootstrap):


  # Estimate MDP and generate data using policy = argmax q_model
  eta = fit_transition_model(env, bootstrap=bootstrap, ixs=train_ixs)
  print('running mb simulations')
  simulation_env = simulate_from_SIS(env, eta, planning_depth, argmaxer, evaluation_budget,
                                     treatment_budget)
  print('estimating q function')
  # Estimate optimal q-function from simulated data
  q_model = rollout(rollout_depth, gamma, simulation_env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                    ixs=train_ixs, bootstrap=False)

  return q_model
