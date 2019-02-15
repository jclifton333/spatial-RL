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
import tensorflow as tf
import datetime
from src.environments.environment_factory import environment_factory
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
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


def fit_q_functions_for_policy(behavior_policy, L, time_horizons, test, iterations=0):
  """
  Generate data under given behavior policy and then evaluate the policy with 0 and 1 step of fitted SARSA.
  :return:
  """
  if test:
    time_horizons = [4, 5]

  time_horizon = int(np.max(time_horizons))

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
    action = behavior_policy(env.X[-1])
    env.step(action)

  # Fit Q-function for myopic policy
  # 0-step Q-function
  q0_dict = {}
  print('Fitting q0s')
  for T in time_horizons:
    y = np.hstack(env.y)
    X = np.vstack(env.X)
    model_name = 'L=100-T={}'.format(T)
    # q0_piecewise = model_fitters.fit_piecewsie_keras_classifier(X, y, np.where(np.vstack(env.X_raw)[:, -1] == 1)[0],
    #                                                             np.where(np.vstack(env.X_raw)[:, -1] == 0)[0],
    #                                                             model_name, test=test)

    clf = model_fitters.SKLogit2()
    clf.fit(X, y, None, False, np.where(np.vstack(env.X_raw)[:, -1] == 1)[0],
            np.where(np.vstack(env.X_raw)[:, -1] == 0)[0])
    q0_piecewise = clf.predict_proba
    q0_dict[T] = q0_piecewise

  if iterations == 1:
    print('Fitting q1')
    # 1-step Q-function
    q0_evaluate_at_pi = np.array([])

    for ix, x in enumerate(env.X[1:]):
      # Get infected and not-infected indices for piecewise predictions
      x_raw = env.X_raw[ix+1]
      infected_indices = np.where(x_raw[:, -1] == 1)[0]
      not_infected_indices = np.where(x_raw[:, -1] == 0)[0]

      a = np.zeros(L)
      probs = q0_piecewise(x, infected_indices, not_infected_indices)
      treat_ixs = np.argsort(-probs)[:treatment_budget]
      a[treat_ixs] = 1
      x_at_a = env.data_block_at_action(ix, a)

      # q0_at_a = q0.predict(x_at_a)
      q0_at_a = q0_piecewise(x_at_a, infected_indices, not_infected_indices)
      q0_evaluate_at_pi = np.append(q0_evaluate_at_pi, q0_at_a)

    X2 = np.vstack(env.X_2[:-1])
    q1_target = np.hstack(env.y[:-1]) + gamma * q0_evaluate_at_pi
    # q1, q1_graph = model_fitters.fit_keras_regressor(X2, q1_target)
    q1_piecewise = model_fitters.fit_piecewise_keras_regressor(X2, q1_target, np.where(np.vstack(env.X_raw[:-1])[:, -1] == 1)[0],
                                                               np.where(np.vstack(env.X_raw[:-1])[:, -1] == 0)[0],
                                                               test=test)

    # return q1, None, env.X_raw, env.X, env.X_2, q1_graph, None
    return q1_piecewise, None, env.X_raw, env.X, env.X_2, None, None
  else:
    # return q0, None, env.X_raw, env.X, env.X_2, q0_graph, None
    return q0_dict, None, env.X_raw, env.X, env.X_2, None, None


def compute_q_function_for_policy_at_state(L, initial_infections, initial_action, behavior_policy,
                                           test):
  """

  :param initial_infections:
  :param initial_action:
  :param policy: Function that takes features in form of X (as opposed to X_raw or X_2) and returns action.
  :return:
  """
  if test:
    MC_REPLICATES = 2
    TIME_HORIZON = 10
  else:
    MC_REPLICATES = 100
    TIME_HORIZON = 50

  gamma = 0.9
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
    r_0 = np.sum(env.current_infected)
    q0_rep += r_0
    q1_rep += r_0
    q_rep += r_0
    for t in range(TIME_HORIZON):
      # env.step(policy.evaluate(env.X[-1]))
      action = behavior_policy(env.X[-1])
      env.step(action)
      r_t = np.sum(env.current_infected)
      q_rep += gamma**t * r_t
      if t < 1:
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
  treatment_budget = int(np.floor(0.05 * L))
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
  # clf = LogisticRegression()
  # clf.fit(X, y)
  # q0_for_behavior_policy = lambda X_: clf.predict_proba(X_)[:, -1]
  # myopic_q_hat_policy_wrapper_ = myopic_q_hat_policy_wrapper(L, q0.predict, treatment_budget)
  # behavior_policy = q0_for_behavior_policy
  # results = {'X_raw': ref_env.X_raw, 'X': ref_env.X, 'X_2': ref_env.X_2, 'behavior_policy': behavior_policy}

  results = {'X_raw': ref_env.X_raw, 'X': ref_env.X, 'X_2': ref_env.X_2, 'behavior_policy': behavior_policy}
  return results


def get_true_q_functions_on_reference_distribution(behavior_policy, L, X_raw, test):
  if test:
    reference_state_indices = range(2)
  else:
    reference_state_indices = range(len(X_raw))

  # Evaluate policy with simulations
  q_true_vals = []
  q_true_ses = []
  q0_true_vals = []
  q1_true_vals = []

  for ix in reference_state_indices:
    print('Computing true q vals at (s, a) {}'.format(ix))
    x_raw = X_raw[ix]

    # Estimate true q function by rolling out policy
    initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
    true_q_at_state, q0_true, q1_true, true_q_se = \
      compute_q_function_for_policy_at_state(L, initial_infections, initial_action,
                                             behavior_policy, test)

    q_true_vals.append(float(true_q_at_state))
    q_true_ses.append(float(true_q_se))
    q0_true_vals.append(float(q0_true))
    q1_true_vals.append(float(q1_true))

  results = {'q_true_vals': q_true_vals, 'q_true_ses': q_true_ses, 'q0_true_vals': q0_true_vals,
             'q1_true_vals': q1_true_vals}

  return results


def compare_fitted_q_to_true_q(X_raw, X, X2, behavior_policy, q0_true, q1_true, q_true, test, time_horizons,
                               fname, L=1000, time_horizon=50, iterations=0):
  """

  :param L:
  :return:
  """
  # Get fitted q, and 0-step q function for policy to be evaluated, and data for reference states
  # X_raw_for_q is the raw data that the q functions were fit on, as oppoosed to the ones where they will be
  # assessed;
  # we use it for tracking the state of the MDP over time.
  qhat0_dict, qhat1, X_raw_for_q, _, _, q0_graph, q1_graph = \
    fit_q_functions_for_policy(behavior_policy, L, time_horizons, test, iterations=iterations)

  # Summarize covariate history
  infection_proportions = [float(np.mean(x[:, -1])) for x in X_raw_for_q]
  state_proportions = [float(np.mean(x[:, 0])) for x in X_raw_for_q]

  if test:
    reference_state_indices = range(2)
  else:
    reference_state_indices = range(len(X))

  results_dict = {'infection_proportions': infection_proportions, 'state_proportions': state_proportions}

  for T, qhat0 in qhat0_dict.items():
    qhat0_vals = []
    qhat1_vals = []
    for ix in reference_state_indices:
      print('Computing estimated q vals at (s, a) {}'.format(ix))
      x = X[ix]

      if iterations == 0:
        x = X[ix]
      elif iterations == 1:
        x = X2[ix]

      x_raw = X_raw[ix]
      infected_indices = np.where(x_raw[:, -1] == 1)[0]
      not_infected_indices = np.where(x_raw[:, -1] == 0)[0]

      # Evaluate 0-step q function
      # qhat0_at_state = np.sum(qhat0.predict(x))
      qhat0_at_state = np.sum(qhat0(x, infected_indices, not_infected_indices))
      qhat0_vals.append(float(qhat0_at_state))

    # Compute rank coefs with true (infinite horizon) q values
    q0_rank_coef = float(spearmanr(q_true, qhat0_vals)[0])
    # q1_rank_coef = float(spearmanr(q_true, qhat1_vals)[0])

    # Compute MSEs with true finite-stage q functions
    if iterations == 0:
      q0_mse = float(np.mean((q0_true - np.array(qhat0_vals))**2))
    elif iterations == 1:
      q0_mse = float(np.mean((q1_true - np.array(qhat0_vals))**2))
    # q1_mse = float(np.mean((q1_true - np.array(qhat1_vals))**2))

    results_dict[T] = {'q0_rank_coef': q0_rank_coef, 'q1_rank_coef': None, 'q0_mse': q0_mse, 'q1_mse': None}
    if not test:
      with open(fname, 'w') as outfile:
        yaml.dump(results_dict, outfile)

  return results_dict


def compare_at_multiple_horizons(L, horizons=(10, 50, 100, 200), test=False, iterations=0):
  # Define behavior policy - it is a myopic policy that treats the locations mostly likely to be infected next,
  # where probabilities are given by expit-linear function of X
  BEHAVIOR_POLICY_COEF = np.array([-0.52, -0.64, -1.22, -1.17, 1.28, 1.3, 0.0, -0.4, 0.1, 0.1, -0.06, -0.04,
                                   0.5, 0.6, 0.2, 0.2])

  def behavior_policy(X):
    actions = np.zeros(L)
    treatment_budget = int(np.floor(0.05 * L))
    logits = np.dot(X, BEHAVIOR_POLICY_COEF)
    locations_to_treat = np.argsort(-logits)[:treatment_budget]
    actions[locations_to_treat] = 1
    return actions

  if test:
    L = 20

  results_dict = {}
  inputs = generate_data_and_behavior_policy(L)
  X_raw, X, X_2 = inputs['X_raw'], inputs['X'], inputs['X_2']

  # Check if there are saved reference state data
  existing_data = False
  for filename in os.listdir('./data_for_prefit_policies/'):
    if 'L={}'.fomrat(L) in filename:
      existing_data = True
      reference_state_data = pkl.load(open(filename, 'r'))

  # If there is no pre-saved data, compute true q-vals on current reference distribution and save
  if not existing_data:
    true_q_vals = get_true_q_functions_on_reference_distribution(behavior_policy, L, X_raw, test)
    reference_state_data = {'X_raw': X_raw, 'X': X, 'X_2': X_2}
    reference_state_data.update(true_q_vals)

  q0_true, q1_true, q_true, q_true_ses = \
    reference_state_data['q0_true_vals'], reference_state_data['q1_true_vals'], reference_state_data['q_true_vals'], \
    reference_state_data['q_true_ses']
  results_dict['q_true_ses'] = q_true_ses

  basename = 'L={}-multiple-horizons-iterations={}'.format(L, iterations)
  time = datetime.datetime.now().strftime("%y%m%d_H%M")
  fname = "{}-{}.yml".format(basename, time)

  compare_fitted_q_to_true_q(reference_state_data['X_raw'], reference_state_data['X'], reference_state_data['X_2'],
                             behavior_policy, q0_true, q1_true,
                             q_true, test, horizons, fname, L=L,
                             iterations=iterations)
  return








