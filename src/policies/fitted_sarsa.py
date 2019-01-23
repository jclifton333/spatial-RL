"""
For a fixed policy, do fitted Q-iteration (i.e. fitted SARSA), and compare with true Q-function to see how well that
Q-function is estimated.  This is for diagnosing issues with so-called  ``Q-learning + policy search''.
"""
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)
import numpy as np
import pdb
from src.environments.environment_factory import environment_factory
import src.environments.generate_network as generate_network
import src.estimation.q_functions.model_fitters as model_fitters
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.q_functions.one_step import fit_one_step_predictor
import numpy as np
from functools import partial
import pickle as pkl
from src.utils.misc import random_argsort
import pprofile

def fit_q_function_for_policy(L, iterations=1):
  """
  Generate data under myopic policy and then evaluate the policy with fitted SARSA.
  :return:
  """
  # Initialize environment
  gamma = 0.9
  treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice})
  env.reset()
  dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
  print('Taking initial steps')
  profiler = pprofile.Profile()
  with profiler:
    env.step(np.random.permutation(dummy_action))
  env.step(np.random.permutation(dummy_action))

  # Rollout using random policy
  print('Rolling out to collect data')
  for t in range(50):
    env.step(np.random.permutation(dummy_action))

  # Fit Q-function for myopic policy
  # 0-step Q-function
  print('Fitting q0')
  y = np.hstack(env.y)
  X = np.vstack(env.X)
  q0 = model_fitters.KerasClassifier()
  q0.fit(X, y.reshape((len(y), 1)), weights=None)
  pdb.set_trace()

  if iterations == 1:
    print('Fitting q1')
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
    q1_target = np.hstack(env.y[:-1]) + gamma * q0_evaluate_at_pi
    q1 = model_fitters.KerasRegressor()
    q1.fit(X2, q1_target, weights=None)
    q_hat = q1.predict
    data_for_q_hat = env.X_2
  elif iterations == 0:
    q_hat = q0.predict
  return q_hat, q0.predict, env.X_raw, data_for_q_hat


def compute_q_function_for_policy_at_state(L, initial_infections, initial_action, policy):
  """

  :param initial_infections:
  :param initial_action:
  :param policy: Function that takes features in form of X (as opposed to X_raw or X_2) and returns action.
  :return:
  """
  gamma = 0.9
  # MC_REPLICATES = 100
  MC_REPLICATES = 10
  # treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
                                      'initial_infections': initial_infections})
  q_list = []
  for rep in range(MC_REPLICATES):
    q_rep = 0.0
    env.reset()
    env.step(initial_action)
    for t in range(50):
      env.step(policy(env.X[-1]))
      q_rep += gamma**t * np.sum(env.current_infected)
    q_list.append(q_rep)
  q = np.mean(q_list)
  se = np.std(q_list) / np.sqrt(MC_REPLICATES)
  return q, se


def compare_fitted_q_to_true_q(L=1000, iterations=1):
  # NUMBER_OF_REFERENCE_STATES = 30
  NUMBER_OF_REFERENCE_STATES = 10
  treatment_budget = int(np.floor(0.05 * L))

  # Get fitted q, and 0-step q function for policy to be evaluated, and data for reference states
  q_hat, q_for_policy, X_raw, data_for_q_hat = fit_q_function_for_policy(L, iterations=iterations)

  def myopic_q_hat_policy(data_block):
    a = np.zeros(L)
    probs = q_for_policy(data_block)
    treat_ixs = random_argsort(probs, treatment_budget)
    a[treat_ixs] = 1
    return a

  # Compare on randomly drawn states
  num_states = len(X_raw)
  reference_state_indices = np.random.choice(num_states, NUMBER_OF_REFERENCE_STATES, replace=False)

  q_hat_vals = np.array([])
  true_q_vals = np.array([])
  true_q_ses = np.array([])
  for rep, ix in enumerate(reference_state_indices):
    print('Computing true and estimated q vals at (s, a) {}'.format(rep))
    x_raw = X_raw[ix]
    x = data_for_q_hat[ix]
    q_hat_at_state = np.sum(q_hat(x))
    q_hat_vals = np.append(q_hat_vals, q_hat_at_state)
    initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
    true_q_at_state, true_q_se = compute_q_function_for_policy_at_state(L, initial_infections, initial_action, myopic_q_hat_policy)
    true_q_vals = np.append(true_q_at_state, true_q_vals)
    true_q_ses = np.append(true_q_ses, true_q_se)

  mean_squared_error = np.mean((true_q_vals - q_hat_vals)**2)

  return mean_squared_error, true_q_vals, true_q_ses, q_hat_vals


if __name__ == "__main__":
  # mse, true_qs, true_q_ses, q_hats = compare_fitted_q_to_true_q(L=100)
  # Initialize environment
  gamma = 0.9
  L = 5000
  treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice})
  env.reset()
  dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
  print('Taking initial steps')
  profiler = pprofile.Profile()
  env.step(np.random.permutation(dummy_action))





































