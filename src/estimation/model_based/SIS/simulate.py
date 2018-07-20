import numpy as np
from src.environments.SIS import SIS
from src.estimation.q_functions.q_functions import q
from src.utils.misc import random_argsort
from functools import partial


def simulate_from_SIS(env, eta, planning_depth, argmaxer, evaluation_budget, treatment_budget,
                      n_rep=30):
  """
  For model-based RL in the SIS generative model.

  :param env: SIS object
  :param eta: length 7 array of disease probability parameters
  :param beta: length 2 array of state transition parameters
  :param planning_depth: how many steps ahead from current state to simulate
  :param argmaxer: dict of policy kwargs
  :param evaluation_budget:
  :param treatment_budget:
  :param n_rep: how many simulation replicates to run
  :return:
  """
  simulation_env = SIS(env.feature_function, env.L, 0, None,
                       adjacency_matrix=env.adjacency_matrix,
                       dict_of_path_lists=env.dict_of_path_lists,
                       initial_infections=env.current_infected,
                       initial_state=env.current_state,
                       eta=eta)
  L = simulation_env.L
  for rep in range(n_rep):
    for t in range(planning_depth):
      # Use myopic estimated probs as rollout (different rollout!) policy
      print('rep {} t {}'.format(rep, t))
      a = np.zeros(L)
      probs = simulation_env.next_infected_probabilities(L)
      treat_ixs = random_argsort(-probs, treatment_budget)
      a[treat_ixs] = 1
      simulation_env.step(a)
    simulation_env.add_state(env.current_state)
    simulation_env.add_infections(env.current_infected)
  return simulation_env
