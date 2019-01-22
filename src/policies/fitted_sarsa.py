"""
For a fixed policy, do fitted Q-iteration (i.e. fitted SARSA), and compare with true Q-function to see how well that
Q-function is estimated.  This is for diagnosing issues with so-called  ``Q-learning + policy search''.
"""
import numpy as np
from src.environments.environment_factory import environment_factory
import src.environments.generate_network as generate_network
import src.estimation.q_functions.model_fitters as model_fitters
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.q_functions.one_step import fit_one_step_predictor
import numpy as np
from functools import partial
import pickle as pkl
from src.utils.misc import random_argsort


def fit_q_function_for_policy():
  """
  Generate data under myopic policy and then evaluate the policy with fitted SARSA.
  :return:
  """
  # Initialize environment
  gamma = 0.9
  L = 1000
  treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice})
  env.reset()
  dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
  env.step(np.random.permutation(dummy_action))
  env.step(np.random.permutation(dummy_action))

  # Rollout using random policy
  for t in range(50):
    env.step(np.random.permutation(dummy_action))

  # Fit Q-function for myopic policy
  # 0-step Q-function
  y = np.hstack(env.y)
  X = np.vstack(env.X)
  q0 = model_fitters.KerasRegressor()
  q0.fit(X, y, weights=None)

  # 1-step Q-function
  q0_evaluate_at_pi = np.array([])

  for ix, x in enumerate(env.X[1:]):
    a = np.zeros(L)
    probs = q0.predict(x)
    treat_ixs = random_argsort(probs, treatment_budget)
    a[treat_ixs] = 1
    x_at_a = env.data_block_at_action(ix, a)
    q0_evaluate_at_pi = np.append(q0_evaluate_at_pi, q0.predict(x_at_a))

  X2 = np.vstack(env.X_2[:-1])
  q1_target = env.y[:-1] + gamma * q0_evaluate_at_pi
  q1 = model_fitters.KerasRegressor()
  q1.fit(X2, q1_target)





















