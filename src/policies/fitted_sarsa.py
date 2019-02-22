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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from src.estimation.optim.argmaxer_factory import argmaxer_quad_approx
import src.environments.generate_network as generate_network
import src.estimation.q_functions.model_fitters as model_fitters
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.environments.sis_infection_probs import sis_infection_probability
import src.environments.sis as sis_helpers
from src.estimation.q_functions.one_step import fit_one_step_sis_mb_q
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from functools import partial
import pickle as pkl
from src.utils.misc import random_argsort
# import pprofile
import multiprocessing as mp
import copy
import yaml
import keras.backend as K
from scipy.special import expit
import argparse


class sklogit3(object):
  def __init__(self, X, y):
   infections = sis_helpers.convert_first_order_to_infection_status(X)
   X = np.column_stack((np.ones(X.shape[0]), X))
   X_times_infection = np.multiply(X, infections[:, np.newaxis])
   X_interaction = np.column_stack((X, X_times_infection))
   clf = LogisticRegression(fit_intercept=False)
   clf.fit(X_interaction, y)
   self.coefficients = clf.coef_[0]

  def predict(self, X_):
    infections = sis_helpers.convert_first_order_to_infection_status(X_)
    X_ = np.column_stack((np.ones(X_.shape[0]), X_))
    Xix = np.multiply(X_, infections[:, np.newaxis])
    Xnew = np.column_stack((X_, Xix))
    prob = expit(np.dot(Xnew, self.coefficients))
    return prob


class q1_rf(object):
  def __init__(self, X, y, q0, gamma, model_name, test=False):
    # self.rf = RandomForestRegressor(n_estimators=100)
    # self.rf.fit(X, y)
    self.reg = model_fitters.fit_piecewise_keras_regressor(X, y, model_name, test=test)
    self.gamma = gamma
    self.q0 = q0

  def predict(self, x1):
    # q0max = self.rf.predict(x1)
    q0max = self.reg.predict(x1).flatten()
    x0 = sis_helpers.convert_second_order_encoding_to_first_order(x1)
    q0 = self.q0(x0)
    return q0 + self.gamma * q0max


class q_mb_wrapper(object):
  """
  Need this bc of STUPID pickling issues.
  """
  def __init__(self, env, L, T):
    # Get model-based
    indices = np.array([[l for l in range(L)] for t in range(T)])
    eta = fit_sis_transition_model(env, indices=indices)
    self.eta = eta
    self.L = L
    self.adjacency_list = env.adjacency_list

  def predict(self, X_raw):
    return sis_infection_probability(X_raw[:, 1], X_raw[:, 2], self.eta, self.L, self.adjacency_list,
                                     **{'s': X_raw[:, 0], 'omega': 0.0})


def fit_optimal_q_functions(L, time_horizons, test, timestamp, iterations=0):
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
    env.step(np.random.permutation(dummy_action))

  # Fit Q-function for myopic policy
  # 0-step Q-function
  q0_mb_dict = {}
  q0_dict = {}
  q0_1_dict = {}
  q0_2_dict = {}
  q1_mb_dict = {}
  q1_dict = {}

  print('Fitting q0s')
  np.random.seed(3)
  indices = [i for i in range(max(time_horizons))]
  np.random.shuffle(indices)
  for T in time_horizons:
    y = np.hstack([env.y[ix] for ix in indices[:T]])
    X = np.vstack([env.X[ix] for ix in indices[:T]])
    model_name_0 = 'L=100-T={}-k=0-{}'.format(T, timestamp)
    # q0_piecewise = model_fitters.fit_piecewise_keras_classifier(X, y, model_name_0, test=test)
    q0_piecewise = sklogit3(X, y)
    q0_dict[T] = q0_piecewise.predict

    # Fit two more models on half the data each double Q_learning for construction of pseudo-outcome
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5)
    q0_1 = sklogit3(X1, y1)
    q0_2 = sklogit3(X2, y2)
    q0_1_dict[T] = q0_1.predict
    q0_2_dict[T] = q0_2.predict

    # Fit one-step model-based as comparison
    q0_mb_wrapper_ = q_mb_wrapper(env, L, T)
    q0_mb_dict[T] = q0_mb_wrapper_.predict

  if iterations == 1:
    for T in time_horizons:
      print('Fitting q1')
      # 1-step Q-function
      q0_evaluate_at_argmax = np.array([])
      q0_evaluate_at_xm1 = np.array([])
      q0_piecewise_T = q0_dict[T]
      q0_1_T = q0_1_dict[T]
      q0_2_T = q0_2_dict[T]

      def q0_1_at_block(a):
        X_at_a = env.data_block_at_action(-1, a)
        q_vals = q0_1_T(X_at_a)
        return q_vals

      def q0_2_at_block(a):
        X_at_a = env.data_block_at_action(-1, a)
        q_vals = q0_2_T(X_at_a)
        return q_vals

      for ix, x in enumerate([env.X[indices[t]+1] for t in range(T-1)]):
        # Get infected and not-infected indices for piecewise predictions
        x_raw = env.X_raw[ix+1]

        # Get optimal actions under the split q-functions
        a_1 = argmaxer_quad_approx(q0_1_at_block, 100, treatment_budget, env)
        a_2 = argmaxer_quad_approx(q0_2_at_block, 100, treatment_budget, env)

        # Evaluate the optimal actions under the other q-function
        q0_1 = q0_2_at_block(a_1)
        q0_2 = q0_1_at_block(a_2)
        q0_pooled = 0.5*(q0_1 + q0_2)

        q0_evaluate_at_argmax = np.append(q0_evaluate_at_argmax, q0_pooled)

      X2 = np.vstack([env.X_2[ix] for ix in indices[:T-1]])
      # q1_target = np.hstack(q0_evaluate_at_xm1) + gamma * q0_evaluate_at_argmax
      q1_target = q0_evaluate_at_argmax  # Only approximate maxQ0(S_tp1); can plug in Q0(S_t) directly
      model_name_1 = 'L=100-T={}-k=1-{}'.format(T, timestamp)
      q1_piecewise = q1_rf(X2, q1_target, q0_piecewise_T, gamma, model_name_1, test=test)
      q1_dict[T] = q1_piecewise.predict
    return q0_dict, q1_dict, env.X_raw, env.X, env.X_2, None, None, q0_mb_dict
  else:
    # return q0, None, env.X_raw, env.X, env.X_2, q0_graph, None
    return q0_dict, None, env.X_raw, env.X, env.X_2, None, None, q0_mb_dict


def evaluate_optimal_qfn_policy_for_single_rep(rep, env, q, iterations, initial_action, time_horizon, treatment_budget,
                                               gamma):
  """
  This will be parallelized across reps. 
  :return: 
  """
  np.random.seed(rep)

  def q_at_block(a):
    x_at_a = env.data_block_at_action(-1, a, neighbor_order=int(iterations + 1))
    q_vals = q(x_at_a)
    return q_vals

  q_rep = 0.0
  q1_rep = 0.0
  env.reset()
  env.step(initial_action)
  r_0 = np.sum(env.current_infected)
  q_rep += r_0
  q1_rep += r_0

  for t in range(time_horizon):
    # env.step(policy.evaluate(env.X[-1]))
    action = argmaxer_quad_approx(q_at_block, 100, treatment_budget, env)
    env.step(action)
    r_t = np.sum(env.current_infected)
    q_rep += gamma**(t+1) * r_t
    if t == 0:
      q1_rep += r_t
    return q_rep, q1_rep


def evaluate_optimal_qfn_policy(q, L, initial_infections, initial_action, test, iterations=0):
  """

  :param initial_infections:
  :param initial_action:
  :return:
  """
  if test:
    MC_REPLICATES = 2
    TIME_HORIZON = 1
  else:
    MC_REPLICATES = 50
    TIME_HORIZON = 20

  gamma = 0.9
  # MC_REPLICATES = num_processes
  env_kwargs = {'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
                                      'initial_infections': initial_infections}
  treatment_budget = int(np.floor(0.05 * L))

  env = environment_factory('sis', **env_kwargs)
  evaluate_at_rep_partial = partial(evaluate_optimal_qfn_policy_for_single_rep, env=env, q=q, iterations=iterations,
                                    initial_action=initial_action, time_horizon=TIME_HORIZON,
                                    treatment_budget=treatment_budget, gamma=gamma)

  # pool = mp.Pool(MC_REPLICATES)
  # q_and_q1_list = pool.map(evaluate_at_rep_partial, range(MC_REPLICATES))
  # pool.terminate()

  q_and_q1_list = []
  for i in range(5):
    res = evaluate_at_rep_partial(i)
    q_and_q1_list.append(res)

  q_list = [q for q, q1 in q_and_q1_list]
  q1_list = [q1 for q, q1 in q_and_q1_list]
  q = np.mean(q_list)
  se = np.std(q_list) / np.sqrt(MC_REPLICATES)
  q1 = np.mean(q1_list)
  se1 = np.std(q1_list) / np.sqrt(MC_REPLICATES)
  return q, se, q1, se1


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
  q1_dict = {}
  print('Fitting q0s')
  for T in time_horizons:
    y = np.hstack(env.y[:T])
    X = np.vstack(env.X[:T])
    model_name_0 = 'L=100-T={}-k=0'.format(T)
    q0_piecewise = model_fitters.fit_piecewsie_keras_classifier(X, y, np.where(np.vstack(env.X_raw[:T])[:, -1] == 1)[0],
                                                                np.where(np.vstack(env.X_raw[:T])[:, -1] == 0)[0],
                                                                model_name_0, test=test)

    # Simple model for debugging purposes
    # clf = model_fitters.SKLogit2()
    # clf.fit(X, y, None, False, np.where(np.vstack(env.X_raw[:T])[:, -1] == 1)[0],
    #         np.where(np.vstack(env.X_raw[:T])[:, -1] == 0)[0])
    # q0_piecewise = clf.predict_proba
    q0_dict[T] = q0_piecewise

  if iterations == 1:
    for T in time_horizons:
      print('Fitting q1')
      # 1-step Q-function
      q0_evaluate_at_pi = np.array([])
      q0_piecewise_T = q0_dict[T]
      for ix, x in enumerate(env.X[1:T]):
        # Get infected and not-infected indices for piecewise predictions
        x_raw = env.X_raw[ix+1]
        infected_indices = np.where(x_raw[:, -1] == 1)[0]
        not_infected_indices = np.where(x_raw[:, -1] == 0)[0]

        # ToDo: Should be using behavior policy!
        a = np.zeros(L)
        probs = q0_piecewise_T(x, infected_indices, not_infected_indices)
        treat_ixs = np.argsort(-probs)[:treatment_budget]
        a[treat_ixs] = 1
        x_at_a = env.data_block_at_action(ix, a)

        # q0_at_a = q0.predict(x_at_a)
        q0_at_a = q0_piecewise_T(x_at_a, infected_indices, not_infected_indices)
        q0_evaluate_at_pi = np.append(q0_evaluate_at_pi, q0_at_a)

      X2 = np.vstack(env.X_2[:T-1])
      q1_target = np.hstack(env.y[:T-1]) + gamma * q0_evaluate_at_pi
      model_name_1 = 'L=100-T={}-k=1'.format(T)
      q1_piecewise = model_fitters.fit_piecewise_keras_regressor(X2, q1_target, np.where(np.vstack(env.X_raw[:T-1])[:, -1] == 1)[0],
                                                                 np.where(np.vstack(env.X_raw[:T-1])[:, -1] == 0)[0],
                                                                 model_name_1,
                                                                 test=test)
      q1_dict[T] = q1_piecewise

    # return q1, None, env.X_raw, env.X, env.X_2, q1_graph, None
    return q0_dict, q1_dict, env.X_raw, env.X, env.X_2, None, None
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
      q_rep += gamma**(t+1) * r_t
      if t < 1:
        q1_rep += gamma * r_t
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

  results = {'X_raw': ref_env.X_raw, 'X': ref_env.X, 'X_2': ref_env.X_2}
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
  qhat0_dict, qhat1_dict, X_raw_for_q, _, _, q0_graph, q1_graph = \
    fit_q_functions_for_policy(behavior_policy, L, time_horizons, test, iterations=iterations)

  # Summarize covariate history
  infection_proportions = [float(np.mean(x[:, -1])) for x in X_raw_for_q]
  state_proportions = [float(np.mean(x[:, 0])) for x in X_raw_for_q]

  if test:
    reference_state_indices = range(2)
  else:
    reference_state_indices = range(len(X))

  results_dict = {'infection_proportions': infection_proportions, 'state_proportions': state_proportions}

  for T in qhat0_dict.keys():
    qhat0 = qhat0_dict[T]
    if iterations == 1:
      qhat1 = qhat1_dict[T]

    qhat0_vals = []
    qhat1_vals = []
    for ix in reference_state_indices:
      print('Computing estimated q vals at (s, a) {}'.format(ix))
      x = X[ix]

      x_raw = X_raw[ix]
      infected_indices = np.where(x_raw[:, -1] == 1)[0]
      not_infected_indices = np.where(x_raw[:, -1] == 0)[0]

      # Evaluate 0-step q function
      # qhat0_at_state = np.sum(qhat0.predict(x))
      qhat0_at_state = np.sum(qhat0(x, infected_indices, not_infected_indices))
      qhat0_vals.append(float(qhat0_at_state))

      if iterations == 1:
        x_2 = X2[ix]
        qhat1_at_state = np.sum(qhat1(x_2, infected_indices, not_infected_indices))
        qhat1_vals.append(float(qhat1_at_state))

    # Compute rank coefs with true (infinite horizon) q values
    q0_rank_coef = float(spearmanr(q_true, qhat0_vals)[0])
    q0_mse = float(np.mean((q0_true - np.array(qhat0_vals))**2))

    if iterations == 1:
      q1_rank_coef = float(spearmanr(q_true, qhat1_vals)[0])
      q1_mse = float(np.mean((q1_true - np.array(qhat1_vals))**2))

    results_dict[T] = {'q0_rank_coef': q0_rank_coef, 'q1_rank_coef': q1_rank_coef, 'q0_mse': q0_mse, 'q1_mse': q1_mse}
    if not test:
      with open(fname, 'w') as outfile:
        yaml.dump(results_dict, outfile)

  return results_dict


def evaluate_qopt_at_multiple_horizons(L, X_raw, X, X2, fname, timestamp, time_horizons=(10, 50, 100, 200), test=False,
                                       iterations=0):
  """

  :param L:
  :return:
  """
  # We need this to compute true probabilities (to assess q0)
  env_kwargs = {'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice}
  ref_env = environment_factory('sis', **env_kwargs)

  qhat0_dict, qhat1_dict, X_raw_for_q, _, _, q0_graph, q1_graph, qhat0_mb_dict = \
    fit_optimal_q_functions(L, time_horizons, test, timestamp, iterations=iterations)

  # Summarize covariate history
  infection_proportions = [float(np.mean(x[:, -1])) for x in X_raw_for_q]
  state_proportions = [float(np.mean(x[:, 0])) for x in X_raw_for_q]

  if test:
    reference_state_indices = range(2)
  else:
    reference_state_indices = range(len(X))

  results_dict = {'infection_proportions': infection_proportions, 'state_proportions': state_proportions}

  for T in qhat0_dict.keys():
    if iterations == 0:
      qhat = qhat0_dict[T]
    elif iterations == 1:
      qhat = qhat1_dict[T]
    qhat_mb = qhat0_mb_dict[T]

    qhat_vals = []
    qhat_mb_vals = []
    qhat_ses = []
    qhat_mb_ses = []
    qhat_mses = []
    qhat_mb_mses = []
    qhat1_estimates = []
    true_q1s = []

    # for ix in reference_state_indices:
    for ix in range(5):  # Only evaluate at one ref state!
      # evaluate_optimal_qfn_policy(qhat1, )
      # print('Computing estimated q vals at (s, a) {}'.format(ix))
      x0 = X[ix]
      x1 = X2[ix]
      x_raw = X_raw[ix]

      a_, y_ = x_raw[:, 1], x_raw[:, 2]
      q, se, q1, se1 = evaluate_optimal_qfn_policy(qhat, L, y_, a_, test,
                                                   iterations=iterations)
      q_mb, se_mb, q1_mb, se1_mb = evaluate_optimal_qfn_policy(qhat_mb, L, y_, a_, test,
                                                               iterations=iterations)
      qhat_vals.append(float(q))
      qhat_mb_vals.append(float(q_mb))
      qhat_ses.append(float(se))
      qhat_mb_ses.append(float(se_mb))

      # Compare to q values
      if iterations == 0:
        kwargs_ = {'omega': 0.0, 's': np.zeros(L)}
        true_probs = sis_infection_probability(a_, y_, ref_env.ETA, L, ref_env.adjacency_list, **kwargs_)
        true_q_mb = true_q = np.sum(true_probs)
        qhat_x = np.sum(qhat(x0))
      elif iterations == 1:
        true_q = q1
        true_q_mb = q1_mb
        qhat_x = np.sum(qhat(x1))
        qhat1_estimates.append(float(qhat_x))
        true_q1s.append(float(q1))
      
      qhat_mb_x = np.sum(qhat_mb(x_raw))
      qhat_mses.append(float((true_q - qhat_x)**2))
      qhat_mb_mses.append(float((true_q_mb - qhat_mb_x)**2))

    results_dict[T] = {'qhat0_vals': qhat_vals, 'qhat0_mean_val': float(np.mean(qhat_vals)),
                       'qhat0_mse': float(np.mean(qhat_mses)), 'qhat0_mb_vals': qhat_mb_vals,
                       'qhat0_mb_mean_val': float(np.mean(qhat_mb_vals)),
                       'qhat0_mb_mse': float(np.mean(qhat_mb_mses)), 'qhat1_estimates': qhat1_estimates,
                        'true_q1s': true_q1s}
    if not test:
      with open(fname, 'w') as outfile:
        yaml.dump(results_dict, outfile)

  return results_dict


def evaluate_qopt(L, horizons=(10, 50, 100, 200), test=False, refit=False, iterations=0):
  # inputs = generate_data_and_behavior_policy(L)

  # Check if there are saved reference state data
  existing_data = False
  if not refit:
    for filename in os.listdir('./data_for_prefit_policies/'):
      if 'L={}'.format(L) in filename:
        existing_data = True
        reference_state_data = pkl.load(open(os.path.join(this_dir, 'data_for_prefit_policies', filename), 'rb'))
  # ToDo: Handle case where there's no existing data

  basename = 'qopt-L={}-iterations={}'.format(L, iterations)
  timestamp = datetime.datetime.now().strftime("%y%m%d_H%M")
  fname = "{}-{}.yml".format(basename, timestamp)
  evaluate_qopt_at_multiple_horizons(L, reference_state_data['X_raw'], reference_state_data['X'],
                                     reference_state_data['X_2'], fname, timestamp, time_horizons=horizons, test=test,
                                     iterations=iterations)
  return


def compare_at_multiple_horizons(L, horizons=(10, 50, 100, 200), test=False, refit=False, iterations=0):
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
  if not refit:
    for filename in os.listdir('./data_for_prefit_policies/'):
      if 'L={}'.format(L) in filename:
        existing_data = True
        reference_state_data = pkl.load(open(os.path.join(this_dir, 'data_for_prefit_policies', filename), 'rb'))

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








