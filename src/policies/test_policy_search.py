import numpy as np
import src.policies.policy_search as ps
from src.environments.environment_factory import environment_factory

# ToDo: Implement transmission probability in sis_infection_probs
from src.environments.sis_infection_probs import sis_infection_probability, sis_transmission_probs_for_omega0
from src.environments.generate_network import lattice


def test_sis():
  env_kwargs = {'L': 25, 'omega': 0, 'generate_network': lattice,
                'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': 0}
  env = environment_factory('sis', **env_kwargs)

  eta_mean = np.random.normal(7)
  eta_cov = np.eye(7)
  initial_policy_parameter = np.ones(3)
  time_horizon = 10
  initial_alpha = initial_zeta = 1
  treatment_budget = 5
  infection_probs_predictor = sis_infection_probability
  transmission_probs_predictor = sis_transmission_probs_for_omega0

  a = ps.policy_search(env, time_horizon, eta_mean, eta_cov, initial_policy_parameter, initial_alpha, initial_zeta,
                       infection_probs_predictor, transmission_probs_predictor, treatment_budget)

  return a




