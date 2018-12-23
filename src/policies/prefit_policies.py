"""
Fit Q-function on (large) batch data and roll out without updating.
"""
import numpy as np
import pickle as pkl
from ..environments.environment_factory import environment_factory
from ..environments.generate_network import lattice


def generate_two_step_sis_data():
  number_of_data_points = 1e6
  L = 100
  time_horizon = 50
  number_of_episodes = int(np.floor(number_of_data_points / (L * time_horizon)))

  X_first_order = []
  X_second_order = []
  y = []
  for _ in number_of_episodes:
    treatment_budget = int(np.floor(0.05 * L))
    env = environment_factory('sis', **{'L': L, 'omega': 0.0 , 'generate_network': lattice})
    env.reset()

    # Initial steps
    dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
    env.step(np.random.permutation(dummy_action))
    env.step(np.random.permutation(dummy_action))
    for t in range(time_horizon - 2):
      env.step(np.random.permutation(dummy_action))
    y.append(np.hstack(env.y))
    X_first_order.append(np.vstack(env.X))
    X_second_order.append(np.vstack(env.X_2))

  X_first_order = np.vstack(X_first_order)
  X_second_order = np.vstack(X_second_order)
  y = np.hstack(y)
  data = {'X_first_order': X_first_order, 'X_second_order': X_second_order, 'y': y}
  fname = './data_for_prefit_policies/two-step-sis.p'
  pkl.dump(data, open(fname, 'wb'))

  return

