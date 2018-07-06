# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np

# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from src.utils.misc import RidgeProb, KerasLogit


def main(lookahead_depth, T, n_rep, env_name, policy_name, argmaxer_name, **kwargs):
  """
  :param lookahead_depth:
  :param env_name: 'SIS' or 'Ebola'
  :param T: duration of simulation rep
  :param n_rep: number of replicates
  :param policy_name: string in ['random', 'no_action', 'true_probs', 'rollout', 'network rollout',
  'one_step'].
  :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
  :param kwargs: environment-specific keyword arguments
  """
  # Initialize generative model
  gamma = 0.9

  def feature_function(x):
    return x

  env = environment_factory(env_name, feature_function, **kwargs)

  # Evaluation limit parameters
  treatment_budget = np.int(np.floor(0.05 * kwargs['L']))
  evaluation_budget = 10

  policy = policy_factory(policy_name)
  random_policy = policy_factory('random') # For initial actions
  argmaxer = argmaxer_factory(argmaxer_name)
  policy_arguments = {'classifier': KerasLogit, 'regressor':RandomForestRegressor, 'env':env,
                      'evaluation_budget':evaluation_budget, 'gamma':gamma, 'rollout_depth':lookahead_depth,
                      'treatment_budget':treatment_budget, 'divide_evenly': False, 'argmaxer': argmaxer}
  score_list = []
  for rep in range(n_rep):
    env.reset()
    env.step(random_policy(**policy_arguments))
    env.step(random_policy(**policy_arguments))
    for t in range(T-2):
      t0 = time.time()
      a = policy(**policy_arguments)
      env.step(a)
      t1 = time.time()
      print('rep: {} t: {} total time: {}'.format(rep, t, t1-t0))
    score_list.append(np.mean(env.Y))
    print('Episode score: {}'.format(np.mean(env.Y)))
  return score_list


if __name__ == '__main__':
  import time
  n_rep = 1
  SIS_kwargs = {'L': 100, 'omega': 1, 'generate_network': generate_network.lattice}
  for k in range(1, 2):
    scores = main(k, 3, n_rep, 'SIS', 'rollout', 'quad_approx', **SIS_kwargs)
    print('k={}: score={} se={}'.format(k, np.mean(scores), np.std(scores) / len(scores)))