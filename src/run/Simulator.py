# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np
import time
import datetime
import yaml
import multiprocessing as mp
import pdb

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
from analysis.bellman_error_bootstrappers import bootstrap_rollout_qfn, bootstrap_SIS_mb_qfn

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from src.utils.misc import RidgeProb, KerasLogit

from functools import partial


class Simulator(object):
  def __init__(self, lookahead_depth, env_name, time_horizon, number_of_replicates, policy_name, argmaxer_name, gamma,
               evaluation_budget, **env_kwargs):
    """
    :param lookahead_depth:
    :param env_name: 'SIS' or 'Ebola'
    :param time_horizon: duration of simulation rep
    :param number_of_replicates: number of replicates
    :param policy_name: string to be passed to policy_factory
    :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
    :param gamma: discount factor
    :param policy_kwargs_dict:
    :param env_kwargs: environment-specific keyword arguments to be passed to environment_factory
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
    treatment_budget = np.int(np.ceil(0.05 * env_kwargs['L']))
    self.policy_arguments =  {'classifier': KerasLogit, 'regressor': RandomForestRegressor, 'env': self.env,
                              'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                              'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                              'divide_evenly': False, 'argmaxer': self.argmaxer, 'q_model': None,
                              'bootstrap': False}

    # Get settings dict for log
    self.settings = {'classifier': self.policy_arguments['classifier'].__name__,
                     'regressor': self.policy_arguments['regressor'].__name__,
                     'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                     'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                     'divide_evenly': self.policy_arguments['divide_evenly'], 'argmaxer': argmaxer_name}
    self.settings.update({'env_name': env_name, 'L': self.env.L, 'policy_name': policy_name,
                          'argmaxer_name': argmaxer_name, 'time_horizon': self.time_horizon,
                          'number_of_replicates': self.number_of_replicates})
    # Currently SIS-specific (omega)
    self.basename = '_'.join([env_name, policy_name, argmaxer_name, str(self.env.L), str(env_kwargs['omega'])])

  def run(self):
    # Multiprocess simulation replicates
    num_processes = int(np.min((self.number_of_replicates, mp.cpu_count() / 3)))
    pool = mp.Pool(processes=num_processes)
    results_list = pool.map(self.episode, range(self.number_of_replicates))

    # Save results
    results_dict = {k: v for d in results_list for k, v in d.items()}
    results_dict['mean'] = float(np.mean([v for v in results_dict.values]))
    self.save_results(results_dict)
    return

  def episode(self, replicate):
    np.random.seed(replicate)
    episode_results = {'score': None, 'runtime': None}
    t0 = time.time()
    self.env.reset()
    # Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    for t in range(self.time_horizon-2):
      a, _ = self.policy(**self.policy_arguments)
      self.policy_arguments['planning_depth'] = self.time_horizon - t
      self.env.step(a)
    t1 = time.time()
    score = np.mean(self.env.Y)
    episode_results['score'] = float(score)
    episode_results['runtime'] = float(t1 - t0)
    return {replicate: episode_results}

  def run_for_profiling(self):
    """
    Run short simulation in order to profile.
    :return:
    """
    pass

  def generate_bootstrap_distributions(self, num_bootstrap_samples):
    classifier, regressor, rollout_depth, gamma, treatment_budget, evaluation_budget, planning_depth = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['rollout_depth'], \
      self.policy_arguments['gamma'], self.policy_arguments['treatment_budget'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['planning_depth']

    mb_be = bootstrap_SIS_mb_qfn(self.env, classifier, regressor, rollout_depth, gamma,
                                 planning_depth, treatment_budget, evaluation_budget, self.argmaxer,
                                 num_bootstrap_samples)
    mf_be = bootstrap_rollout_qfn(self.env, classifier, regressor, rollout_depth, gamma,
                                  treatment_budget, evaluation_budget, self.argmaxer, num_bootstrap_samples)
    return mb_be, mf_be

  def bootstrap_distribution_episode(self, num_bootstrap_samples, times_to_evaluate=[0, 1, 3, 5, 10, 15, 20]):

    bootstrap_results = {'mb_be': [], 'mf_be': []}
    self.env.reset()
    # Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])

    for t in range(self.time_horizon - 2):
      a, _ = self.random_policy(**self.policy_arguments)
      self.policy_arguments['planning_depth'] = self.time_horizon - t
      self.env.step(a)
      if t in times_to_evaluate:
        mb_be, mf_be = self.generate_bootstrap_distributions(num_bootstrap_samples)
        bootstrap_results['mb_be'].append(mb_be)
        bootstrap_results['mf_be'].append(mf_be)
    return bootstrap_results

  def bootstrap_distribution_episode_wrapper(self, replicate):
    """
    Wrap for multiprocessing.
    :return: """
    np.random.seed(replicate)
    results = self.bootstrap_distribution_episode(self.num_bootstrap_samples, self.times_to_evaluate)
    return {replicate: results}

  def run_generate_bootstrap_distributions(self, num_bootstrap_samples=30, times_to_evaluate=[0, 1, 3, 5, 10, 15, 20]):
    """
    :param num_bootstrap_samples:
    :param times_to_evaluate: Times at which to generate bootstrap samples.
    :return:
    """
    # Augment save info with bootstrap-specific stuff
    self.basename = '_'.join(['bootstrap_dbns', self.basename])
    self.settings['times_to_evaluate'] = times_to_evaluate
    self.times_to_evaluate = times_to_evaluate
    self.num_bootstrap_samples = num_bootstrap_samples

    # Multiprocess simulation replicates
    num_processes = int(np.min((self.number_of_replicates, mp.cpu_count() / 3)))
    pool = mp.Pool(processes=num_processes)
    results_list = pool.map(self.bootstrap_distribution_episode_wrapper, range(self.number_of_replicates))

    # Save results
    results_dict = {k: v for d in results_list for k, v in d.items()}
    self.save_results(results_dict)

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return
