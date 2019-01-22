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

  return q1.predict, q0.predict, env.X_raw, env.X_2


def compute_q_function_for_policy_at_state(initial_infections, initial_action, policy):
  """

  :param initial_infections:
  :param initial_action:
  :param policy: Function that takes features in form of X (as opposed to X_raw or X_2) and returns action.
  :return:
  """
  gamma = 0.9
  MC_REPLICATES = 10000
  L = 1000
  treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
                                      'initial_infections': initial_infections})
  q = 0.0
  for rep in range(MC_REPLICATES):
    q_rep = 0.0
    env.reset()
    env.step(initial_action)
    for t in range(50):
      env.step(policy(env.X[-1]))
      q_rep += gamma**t * np.sum(env.current_infected)
    q += (q_rep - q) / (rep + 1)
  return q


def compare_fitted_q_to_true_q():
  NUMBER_OF_REFERENCE_STATES = 100
  treatment_budget = 50

  # Get fitted q, and 0-step q function for policy to be evaluated, and data for reference states
  q_hat, q_for_policy, X_raw, X_2 = fit_q_function_for_policy()
  def myopic_q_hat_policy(data_block):
    a = np.zeros(treatment_budget)
    probs = q_for_policy(data_block)
    treat_ixs = random_argsort(probs, treatment_budget)
    a[treat_ixs] = 1
    return a

  # Compare on randomly drawn states
  num_states = len(X_raw)
  reference_state_indices = np.random.choice(num_states, NUMBER_OF_REFERENCE_STATES, replace=False)

  for ix in reference_state_indices:
    x_raw = X_raw[ix]
    x_2 = X_2[ix]
    q_hat_at_state = np.sum(q_hat(x_2))
    initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
    true_q_at_state = compute_q_function_for_policy_at_state(initial_infections, initial_action, myopic_q_hat_policy)



































