import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

import numpy as np
import src.policies.policy_search as ps
from src.environments.environment_factory import environment_factory

from src.environments.sis_infection_probs import sis_infection_probability, get_all_sis_transmission_probs_omega0
from src.environments.generate_network import lattice
import numpy as np


def tune_stochastic_approximation():
  # See supplementary materials of RSS paper (ctrl + f ''tuning procedure'')
  env_kwargs = {'L': 25, 'omega': 0, 'generate_network': lattice,
                'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': 0}
  env = environment_factory('sis', **env_kwargs)
  B = 100
  T = 15  # Different T from time horizon!  Trying to be consistent with supplementary materials.
  RHO_GRID = np.linspace(0.1, 5, num=10)  # ToDo: How to choose these?
  TAU_GRID = np.linspace(0.1, 5, num=10)
  DELTA = [(rho, tau) for rho in RHO_GRID for tau in TAU_GRID]

  qhat_deltas = []
  for rho, tau in DELTA:
    qhat_delta = 0.0
    for b in range(B):
      # Initialize state and history
      env.reset()
      for r in range(T):
        policy_kwargs = {'env': env, 'planning_depth': T, 'treatment_budget': np.int(np.floor(0.05 * env.L)),
                         'rho': rho, 'tau': tau}
        a = ps.policy_search_policy(**policy_kwargs)
        env.step(a)
      qhat_delta += np.mean(env.Y)
    print('rho: {} tau: {} qhat: {}'.format(rho, tau, qhat_delta))
    qhat_deltas.append(qhat_delta)

  best_qhat_ix = np.argmax(qhat_deltas)
  best_params = DELTA[best_qhat_ix]
  return best_params


if __name__ == '__main__':
  tune_stochastic_approximation()