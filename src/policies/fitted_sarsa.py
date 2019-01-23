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
import pprofile
import multiprocessing as mp
import yaml


def fit_q_functions_for_policy(L):
  """
  Generate data under myopic policy and then evaluate the policy with 0 and 1 step of fitted SARSA.
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
  for t in range(50):
    env.step(np.random.permutation(dummy_action))

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


def compute_q_function_for_policy_at_state(L, initial_infections, initial_action, policy, num_processes=2):
  """

  :param initial_infections:
  :param initial_action:
  :param policy: Function that takes features in form of X (as opposed to X_raw or X_2) and returns action.
  :return:
  """
  gamma = 0.9
  # MC_REPLICATES = 100
  MC_REPLICATES = num_processes
  env_kwargs = {'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
                                      'initial_infections': initial_infections}

  # env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': generate_network.lattice,
  #                                     'initial_infections': initial_infections})
  # for rep in range(MC_REPLICATES):
  #   q_rep = 0.0
  #   env.reset()
  #   env.step(initial_action)
  #   for t in range(50):
  #     env.step(policy(env.X[-1]))
  #     q_rep += gamma**t * np.sum(env.current_infected)
  #   q_list.append(q_rep)

  def replicate_wrapper(replicate_index):
    np.random.seed(replicate_index)
    q_rep_ = single_replicate_of_compute_q_function_at_state(env_kwargs, gamma, initial_action, policy)
    return q_rep_

  pool = mp.Pool(num_processes)
  q_list = pool.map(replicate_wrapper, range(MC_REPLICATES))
  q = np.mean(q_list)
  se = np.std(q_list) / np.sqrt(MC_REPLICATES)
  return q, se


def single_replicate_of_compute_q_function_at_state(env_kwargs, gamma, initial_action, policy):
  q_rep = 0.0
  env = environment_factory('sis', env_kwargs)
  env.reset()
  env.step(initial_action)
  for t in range(50):
      env.step(policy(env.X[-1]))
      q_rep += gamma**t * np.sum(env.current_infected)
  return q_rep


def compute_estimated_and_true_qs_at_state(x_tuple, L, qhat0, qhat1, myopic_q_hat_policy):
  """

  :param x: tuple (x_raw, x, x2)
  :param L:
  :param qhat0:
  :param qhat1:
  :param myopic_q_hat_policy:
  :return:
  """
  x_raw, x, x2 = x_tuple

  # Evaluate 0-step q function
  qhat0_at_state = np.sum(qhat0(x))

  # Evaluate 1-step q function
  qhat1_at_state = np.sum(qhat1(x2))

  # Estimate true q function by rolling out policy
  initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
  true_q_at_state, true_q_se = compute_q_function_for_policy_at_state(L, initial_infections, initial_action,
                                                                      myopic_q_hat_policy)
  return {'qhat0_at_state': float(qhat0_at_state), 'qhat1_at_state': float(qhat1_at_state),
          'true_q_at_state': float(true_q_at_state), 'true_q_se': float(true_q_se)}


def compare_fitted_q_to_true_q(L=1000, num_processes=2):
  """

  :param L:
  :return:
  """
  NUMBER_OF_REFERENCE_STATES = num_processes
  # NUMBER_OF_REFERENCE_STATES = 20
  treatment_budget = int(np.floor(0.05 * L))

  # Get fitted q, and 0-step q function for policy to be evaluated, and data for reference states
  qhat0, qhat1, X_raw, X, X2 = fit_q_functions_for_policy(L)

  def myopic_q_hat_policy(data_block):
    a = np.zeros(L)
    probs = qhat0(data_block)
    treat_ixs = np.argsort(-probs)[:treatment_budget]
    a[treat_ixs] = 1
    return a

  # Compare on randomly drawn states
  num_states = len(X_raw)
  reference_state_indices = np.random.choice(num_states, NUMBER_OF_REFERENCE_STATES, replace=False)

  # qhat0_vals = []
  # qhat1_vals = []
  # true_q_vals = []
  # true_q_ses = []

  # for rep, ix in enumerate(reference_state_indices):
  #   print('Computing true and estimated q vals at (s, a) {}'.format(rep))
  #   x_raw = X_raw[ix]
  #   x = X[ix]
  #   x2 = X2[ix]

  #   # Evaluate 0-step q function
  #   qhat0_at_state = np.sum(qhat0(x))
  #   qhat0_vals.append(float(qhat0_at_state))

  #   # Evaluate 1-step q function
  #   qhat1_at_state = np.sum(qhat1(x2))
  #   qhat1_vals.append(float(qhat1_at_state))

  #   # Estimate true q function by rolling out policy
  #   initial_action, initial_infections = x_raw[:, 1], x_raw[:, 2]
  #   true_q_at_state, true_q_se = compute_q_function_for_policy_at_state(L, initial_infections, initial_action,
  #                                                                       myopic_q_hat_policy)
  #   true_q_vals.append(float(true_q_at_state))
  #   true_q_ses.append(float(true_q_se))

  # Evaluate q functions at reference states in parallel
  X_tuples = [(X_raw[ix], X[ix], X2[ix]) for ix in reference_state_indices]
  evaluate_at_state_partial = partial(compute_estimated_and_true_qs_at_state, L, qhat0, qhat1, myopic_q_hat_policy)
  pool = mp.Pool(num_processes)
  pool_results = pool.map(evaluate_at_state_partial, X_tuples)

  # Collect results
  true_q_vals = [r['true_q_at_state'] for r in pool_results]
  qhat0_vals = [r['qhat0_at_state'] for r in pool_results]
  qhat1_vals = [r['qhat1_at_state'] for r in pool_results]
  true_q_ses = [r['true_q_se'] for r in pool_results]

  # Posterior dbn of rank coefficients between (0) q0 and estimated true q and (1) q1 and estimated true q.
  true_q_draws = np.random.multivariate_normal(mean=true_q_vals, cov=np.diag(true_q_ses), size=100)
  q0_rank_coef_draws = [float(spearmanr(true_q, qhat0_vals)[0]) for true_q in true_q_draws]
  q1_rank_coef_draws = [float(spearmanr(true_q, qhat1_vals)[0]) for true_q in true_q_draws]

  q0_rank_coef_mean, q0_rank_coef_se = float(np.mean(q0_rank_coef_draws)), \
                                       float(np.std(q0_rank_coef_draws) / np.sqrt(len(q0_rank_coef_draws)))
  q1_rank_coef_mean, q1_rank_coef_se = float(np.mean(q1_rank_coef_draws)), \
                                       float(np.std(q1_rank_coef_draws) / np.sqrt(len(q1_rank_coef_draws)))

  print('q0 rank mean: {} se: {}'.format(q0_rank_coef_mean, q0_rank_coef_se))
  print('q1 rank mean: {} se: {}'.format(q1_rank_coef_mean, q1_rank_coef_se))

  results = {'true_q_vals': true_q_vals, 'true_q_ses': true_q_ses, 'qhat0_vals': qhat0_vals, 'qhat1_vals': qhat1_vals,
             'q0_rank_coef_mean': q0_rank_coef_mean, 'q0_rank_coef_se': q0_rank_coef_se,
             'q1_rank_coef_mean': q1_rank_coef_mean, 'q1_rank_coef_se': q1_rank_coef_se}

  return results


if __name__ == "__main__":
  results_L100 = compare_fitted_q_to_true_q(L=100)
  results_L1000 = compare_fitted_q_to_true_q(L=1000)

  # Save results to yml
  final_results = {100: results_L100, 1000: results_L1000}
  filename = 'compare-q0-with-q1.yml'
  with open(filename, 'w') as outfile:
    yaml.dump(final_results, outfile)






































