"""
Compare behavior of argmaxer at estimated probabilities to that at true probabilities.
"""
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
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
import src.policies.diagnostics.fitted_sarsa as fs


def fit_and_take_max_for_multiple_draws(L, time_horizon, number_of_tries=50):
  # Initialize environment
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

  q0_mb_wrapper = fs.q_mb_wrapper(env, L, time_horizon)
  maxes_at_each_state = []
  for t in range(5):
    def q0_mb_at_t(a):
      x = env.data_block_at_action(t, a, raw=True)
      phat = q0_mb_wrapper.predict(x)
      return phat

    def q0_at_t(a):
      y = env.Y[t, :]
      p = sis_infection_probability(a, y, env.ETA, env.L, env.adjacency_list, **{'s': np.zeros(L), 'omega': 0.0})
      return p

    # phat = q0_mb_wrapper.predict(env.X_raw[t])

    # p = sis_infection_probability(env.X_raw[t][:, 1], env.X_raw[t][:, 2], env.ETA, env.L, env.adjacency_list,
    #                                **{'s': np.zeros(L), 'omega': 0.0})
    # phats = np.append(phats, phat)
    # ps = np.append(ps, p)

    maxes = []
    max_at_each_draw = []
    for draws in range(number_of_tries):
      a = argmaxer_quad_approx(q0_at_t, 100, treatment_budget, env)
      max_phat = q0_at_t(a) 
      max_q = np.sum(max_phat)
      max_at_each_draw.append(max_q)
      maxes.append(np.min(max_at_each_draw))
    maxes_at_each_state.append(maxes)

  return maxes_at_each_state


def fit_and_take_max(L, time_horizons, test):
  if test:
    time_horizons = [3]

  time_horizon = int(np.max(time_horizons))

  # Initialize environment
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

  q0_mb_dict = {}

  # Estimate probabilities using data up to each time horizon
  for T in time_horizons:
    np.random.seed(T)
    y = np.hstack(env.y[:T])
    X = np.vstack(env.X[:T])

    q0_mb_wrapper = fs.q_mb_wrapper(env, L, T)
    q0_mb_dict[T] = q0_mb_wrapper.predict

    phats = np.array([])
    ps = np.array([])
    phat_maxes = np.array([])
    p_maxes = np.array([])
    qhat_maxes = np.array([])
    q_maxes = np.array([])

    # Compare argmax of estimate to argmax of true probs at each state
    for t in range(int(np.min(time_horizons))):
      def q0_mb_at_t(a):
        x = env.data_block_at_action(t, a, raw=True)
        phat = q0_mb_wrapper.predict(x)
        return phat

      def q0_at_t(a):
        y = env.Y[t, :]
        p = sis_infection_probability(a, y, env.ETA, env.L, env.adjacency_list, **{'s': np.zeros(L), 'omega': 0.0})
        return p

      phat = q0_mb_wrapper.predict(env.X_raw[t])
      p = sis_infection_probability(env.X_raw[t][:, 1], env.X_raw[t][:, 2], env.ETA, env.L, env.adjacency_list,
                                     **{'s': np.zeros(L), 'omega': 0.0})
      phats = np.append(phats, phat)
      ps = np.append(ps, p)

      max_phats = []
      for i in range(3):
        argmax_phat = argmaxer_quad_approx(q0_mb_at_t, 100, treatment_budget, env)
        max_phats.append(q0_mb_at_t(argmax_phat))
      max_phat = np.array(max_phats).mean(axis=0)
      # argmax_phat = argmaxer_quad_approx(q0_at_t, 100, treatment_budget, env)
      # max_phat = q0_at_t(argmax_phat)
      argmax_p = argmaxer_quad_approx(q0_at_t, 100, treatment_budget, env)
      max_p = q0_at_t(argmax_p)
      phat_maxes = np.append(phat_maxes, max_phat)
      p_maxes = np.append(p_maxes, max_p)
      qhat_max = np.sum(max_phat)
      q_max = np.sum(max_p)
      qhat_maxes = np.append(qhat_maxes, qhat_max)
      q_maxes = np.append(q_maxes, q_max)
      print('qmax1: {} qmax2: {}'.format(qhat_max, q_max))

    phat_mse = np.mean((phats - ps)**2)
    phat_max_mse = np.mean((phat_maxes - p_maxes)**2)
    phat_max_bias = np.mean((phat_maxes - p_maxes))
    qhat_max_mse = np.mean((q_maxes - qhat_maxes)**2)
    qhat_max_bias = np.mean((qhat_maxes - q_maxes))
    print('Horizon={} phat mse: {} phat max mse: {} phat max bias: {} qhat max mse: {} qhat max bias: {}'.format(T, phat_mse, phat_max_mse, phat_max_bias,
      qhat_max_mse, qhat_max_bias))

  return


if __name__ == "__main__":
  # maxes = fit_and_take_max_for_multiple_draws(100, 10, number_of_tries=20)
  # for m in maxes: print(m)
  np.random.seed(3)
  fit_and_take_max(100, [10, 50, 100, 200], test=False)


