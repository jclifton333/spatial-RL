import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

import numpy as np
import src.policies.policy_search as ps
from src.environments.environment_factory import environment_factory

# ToDo: Implement transmission probability in sis_infection_probs
from src.environments.sis_infection_probs import sis_infection_probability, get_all_sis_transmission_probs_omega0
from src.environments.generate_network import lattice


def test_sis():
  env_kwargs = {'L': 25, 'omega': 0, 'generate_network': lattice,
                'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': 0}
  env = environment_factory('sis', **env_kwargs)

  eta_mean = np.zeros(7)
  eta_cov = np.eye(7)
  initial_policy_parameter = np.ones(3)
  time_horizon = 25
  initial_alpha = initial_zeta = 1
  treatment_budget = 5
  infection_probs_predictor = sis_infection_probability
  transmission_probs_predictor = get_all_sis_transmission_probs_omega0

  def gen_model_posterior():
    return np.random.multivariate_normal(eta_mean, eta_cov)

  a = ps.policy_search(env, time_horizon, gen_model_posterior, initial_policy_parameter, initial_alpha, initial_zeta,
                       infection_probs_predictor, transmission_probs_predictor, treatment_budget, 1, 1)

  return a


if __name__ == '__main__':
  test_sis()



