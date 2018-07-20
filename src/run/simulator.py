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


class Simulator(object):
  def __init__(self, lookahead_depth, env_name, time_horizon, number_of_replicates, policy_name, argmaxer_name, gamma,
               **env_kwargs):
    """
    :param lookahead_depth:
    :param env_name: 'SIS' or 'Ebola'
    :param time_horizon: duration of simulation rep
    :param number_of_replicates: number of replicates
    :param policy_name: string to be passed to policy_factory
    :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
    :param kwargs: environment-specific keyword arguments
    """
    self.env = environment_factory(env_name, **env_kwargs)
    self.policy = policy_factory(policy_name)
    self.random_policy = policy_factory('random')
    self.argmaxer = argmaxer_factory(argmaxer_name)
    self.time_horizon = time_horizon
    self.number_of_replicates = number_of_replicates
    self.scores = []
    self.runtimes = []

    # Set policy arguments
    treatment_budget = np.int(np.floor(0.05 * env_kwargs['L']))
    evaluation_budget = 10
    self.policy_arguments =  {'classifier': KerasLogit, 'regressor': RandomForestRegressor, 'env': self.env,
                              'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                              'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                              'divide_evenly': False, 'argmaxer': self.argmaxer, 'q_model': None}

    # Get settings dict for log
    self.settings = {'policy_arguments': self.policy_arguments}
    self.settings.update({'env_name': env_name, 'env_kwargs': env_kwargs, 'policy_name': policy_name,
                          'argmaxer_name': argmaxer_name, 'time_horizon': self.time_horizon,
                          'number_of_replicates': self.number_of_replicates})
    self.basename = '_'.join([env_name, policy_name, argmaxer_name, self.env.L])

  def run(self):
    for rep in range(self.number_of_replicates):
      t0 = time.time()
      self.env.reset()
      # Initial steps
      self.env.step(self.random_policy(**self.policy_arguments)[0])
      self.env.step(self.random_policy(**self.policy_arguments)[0])
      for t in range(self.time_horizon-2):
        a, q_model = self.policy(**self.policy_arguments)
        self.policy_arguments['planning_depth'] = self.time_horizon - t
        self.policy_arguments['q_model'] = q_model
        self.env.step(a)
      t1 = time.time()
      self.scores.append(np.mean(self.env.Y))
      self.runtimes.append(t1 - t0)

  def run_for_profiling(self):
    """
    Run short simulation in order to profile.
    :return:
    """
    pass

  def save_results(self):
    pass
