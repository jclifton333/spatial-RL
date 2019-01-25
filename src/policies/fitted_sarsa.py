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
from scipy.stats import spearmanr
import src.environments.generate_network as generate_network
import src.estimation.q_functions.model_fitters as model_fitters
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.q_functions.one_step import fit_one_step_predictor
import numpy as np
from functools import partial
import pickle as pkl
from src.utils.misc import random_argsort
# import pprofile
import multiprocessing as mp
import yaml
import keras.backend as K
import argparse


def fit_q_functions_for_policy(behavior_policy, L, time_horizon):
  """
  Generate data under given behavior policy and then evaluate the policy with 0 and 1 step of fitted SARSA.
  :return:
  """
  # Initialize environment
  gamma = 0.9
  treatment_budget = int(np.floor(0.05 * L))
  env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice})
  env.reset()
  dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
  print('Taking initial steps')
  env.step(np.random.permutation(dummy_action))
  env.step(np.random.permutation(dummy_action))

  # Rollout using random policy
  print('Rolling out to collect data')
  for t in range(time_horizon):
    env.step(behavior_policy(env.X[-1]))

  # Fit Q-function for myopic policy
  # 0-step Q-function
  print('Fitting q0')
  y = np.hstack(env.y)
  X = np.vstack(env.X)
  q0 = model_fitters.fit_keras_classifier(X, y)

  print('Fitting q1')
  # 1-step Q-function
  q0_evaluate_at_pi = np.array([])

  for ix, x in enumerate(env.X[1:]):
    a = np.zeros(L)
    probs = q0.predict(x)
    treat_ixs = np.argsort(-probs)[:treatment_budget]
    a[treat_ixs] = 1
    x_at_a = env.data_block_at_action(ix, a)
    q0_evaluate_at_pi = np.append(q0_evaluate_at_pi, q0.predict(x_at_a))

  X2 = np.vstack(env.X_2[:-1])
  q1_target = np.hstack(env.y[:-1]) + gamma * q0_evaluate_at_pi
  q1 = model_fitters.fit_keras_regressor(X2, q1_target)

  return q0.predict, q1.predict, env.X_raw, env.X, env.X_2


def compute_q_function_for_policy_at_state(L, initial_infections, initial_action, policy):
  """

  :param initial_infections:
  :param initial_action:
  :param policy: Function that takes features in form of X (as opposed to X_raw or X_2) and returns action.
  :return:
  """
  gamma = 0.9
  MC_REPLICATES = 100
  # MC_REPLICATES = num_processes
  env_kwargs = {'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
                                      'initial_infections': initial_infections}

  env = environment_factory('sis', **env_kwargs)
  q_list = []
  q0_list = []
  q1_list = []
  for rep in range(MC_REPLICATES):
    q_rep = 0.0
    q0_rep = 0.0
    q1_rep = 0.0
    env.reset()
    env.step(initial_action)
    for t in range(50):
      env.step(policy.evaluate(env.X[-1]))
      r_t = np.sum(env.current_infected)
      q_rep += gamma**t * r_t
      if t < 1:
        q0_rep += r_t
      if t < 2:
        q1_rep += gamma**t * r_t
    q_list.append(q_rep)
    q0_list.append(q0_rep)
    q1_list.append(q1_rep)

  q = np.mean(q_list)
  q0 = np.mean(q0_list)
  q1 = np.mean(q1_list)
  se = np.std(q_list) / np.sqrt(MC_REPLICATES)

  return q, q0, q1, se


class myopic_q_hat_policy_wrapper(object):
  """
  To get around issues with pickling nested functions...
  """
  def __init__(self, L, qhat0, treatment_budget):
    self.L = L
    self.qhat0 = qhat0
    self.treatment_budget = treatment_budget

  def evaluate(self, data_block):
    a = np.zeros(self.L)
    probs = self.qhat0(data_block)
    treat_ixs = np.argsort(-probs)[:self.treatment_budget]
    a[treat_ixs] = 1
    return a


def generate_data_and_behavior_policy(L=100):
  # Generate reference states
  L = 100
  treatment_budget = int(np.floor(0.5 * L))
  env_kwargs = {'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice}
  ref_env = environment_factory('sis', **env_kwargs)
  dummy_action = np.concatenate((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
  for t in range(100):
    ref_env.step(np.random.permutation(dummy_action))

  # Get behavior policy to be evaluated
  # Fit Q-function for myopic policy
  # 0-step Q-function
  print('Fitting q0')
  y = np.hstack(ref_env.y)
  X = np.vstack(ref_env.X)
  q0 = model_fitters.fit_keras_classifier(X, y)
  myopic_q_hat_policy_wrapper_ = myopic_q_hat_policy_wrapper(L, q0.predict, treatment_budget)

  results = {'X_raw': env.X_raw, 'X': X, 'X_2': ref_env.X_2, 'behavior_policy': myopic_q_hat_policy_wrapper_}
  return results


def get_true_q_functions_on_reference_distribution(behavior_policy, L, X_raw, test):
  if test:
    reference_state_indices = np.random.choice(len(X_raw), 2)
  else:
    reference_state_indices = range(len(X_raw))


  for ix in reference_state_indices:
  # Evaluate policy with simulations
  q_true_vals = []
  q_true_ses = []
  q0_true_vals = []
  q1_true_vals = []

  for ix in range(len(X)):
    print('Computing true q vals at (s, a) {}'.format(ix))
    x_raw = X_raw[ix]

    # Estimate true q function by rolling out policy
    initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
    true_q_at_state, q0_true, q1_true, true_q_se = \
      compute_q_function_for_policy_at_state(L, initial_infections, initial_action,
                                             behavior_policy)

    q_true_vals.append(float(true_q_at_state))
    q_true_ses.append(float(true_q_se))
    q0_true_vals.append(float(q0_true))
    q1_true_vals.append(float(q1_true))

  results = {'q_true_vals': q_true_vals, 'q_true_ses': q_true_ses, 'q0_true_vals': q0_true_vals,
             'q1_true_vals': q1_true_vals}

  return results


def compare_fitted_q_to_true_q(X_raw, X, X2, behavior_policy, q0_true, q1_true, q_true, L=1000, time_horizon=50):
  """

  :param L:
  :return:
  """
  # Get fitted q, and 0-step q function for policy to be evaluated, and data for reference states
  qhat0, qhat1, _, _, _ = fit_q_functions_for_policy(behavior_policy, L, time_horizon)

  qhat0_vals = []
  qhat1_vals = []

  for ix in range(len(X)):
    print('Computing estimated q vals at (s, a) {}'.format(ix))
    x_raw = X_raw[ix]
    x = X[ix]
    x2 = X2[ix]

    # Evaluate 0-step q function
    qhat0_at_state = np.sum(qhat0(x))
    qhat0_vals.append(float(qhat0_at_state))

    # Evaluate 1-step q function
    qhat1_at_state = np.sum(qhat1(x2))
    qhat1_vals.append(float(qhat1_at_state))

  K.clear_session()  # Done with neural nets

  # Compute rank coefs with true (infinite horizon) q values
  q0_rank_coef = float(spearmanr(q_true, qhat0_vals)[0])
  q1_rank_coef = float(spearmanr(q_true, qhat1_vals)[0])

  # Compute MSEs with true finite-stage q functions
  q0_mse = float(np.mean((q0_true - np.array(qhat0_vals))**2))
  q1_mse = float(np.mean((q1_true - np.array(qhat1_vals))**2))

  results = {'q0_rank_coef': q0_rank_coef, 'q1_rank_coef': q1_rank_coef, 'q0_mse': q0_mse, 'q1_mse': q1_mse}

  return results


def compare_at_multiple_horizons(L, horizons=(10, 30, 50, 70, 90), test=False):
  inputs = generate_data_and_behavior_policy(L)
  results_dict = {}
  X_raw, X, X_2, behavior_policy = inputs['X_raw'], inputs['X'], inputs['X_2'], inputs['behavior_policy']
  true_q_vals = get_true_q_functions_on_reference_distribution(behavior_policy, L, X_raw, test)
  q0_true, q1_true, q_true = true_q_vals['q0_true_vals'], true_q_vals['q1_true_vals'], true_q_vals['q_true_vals']

  for time_horizon in horizons:
    results = compare_fitted_q_to_true_q(X_raw, X, X_2, behavior_policy, q0_true, q1_true, q_true, L=L,
                                         time_horizon=time_horizon)
    results_dict[time_horizon] = results

    if not test:
      with open('L={}-multiple-horizons.yml'.format(L), 'w') as outfile:
        yaml.dump(results_dict, outfile)

  return








