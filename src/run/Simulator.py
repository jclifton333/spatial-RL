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

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.model_based.sis.simulate_from_sis import simulate_from_SIS
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory
from src.estimation.stacking.bellman_error_bootstrappers import bootstrap_rollout_qfn, bootstrap_SIS_mb_qfn
from scipy.stats import normaltest, ks_2samp
import copy

from src.estimation.q_functions.model_fitters import KerasRegressor, SKLogit, SKLogit2
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

import keras.backend as K


class Simulator(object):
  def __init__(self, lookahead_depth, env_name, time_horizon, number_of_replicates, policy_name, argmaxer_name, gamma,
               evaluation_budget, env_kwargs, network_name, bootstrap, seed, error_quantile,
               sampling_dbn_run=False, sampling_dbn_estimator=None, fit_qfn_at_end=False,
    ignore_errors=False):
    """
    :param lookahead_depth:
    :param env_name: 'sis' or 'Gravity'
    :param time_horizon: duration of simulation rep
    :param number_of_replicates: number of replicates
    :param policy_name: string to be passed to policy_factory
    :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
    :param gamma: discount factor
    :param policy_kwargs_dict:
    :param env_kwargs: environment-specific keyword arguments to be passed to environment_factory
    """
    self.env_name = env_name
    self.env_kwargs = env_kwargs
    self.env = environment_factory(env_name, **env_kwargs)
    self.policy = policy_factory(policy_name)
    self.random_policy = policy_factory('random')
    self.argmaxer = argmaxer_factory(argmaxer_name)
    self.time_horizon = time_horizon
    self.number_of_replicates = number_of_replicates
    self.ignore_errors = ignore_errors
    self.scores = []
    self.runtimes = []
    self.seed = seed
    self.fit_qfn_at_end = fit_qfn_at_end
    self.sampling_dbn_estimator = sampling_dbn_estimator

    # Set policy arguments
    if env_name in ['sis', 'ContinuousGrav']:
        treatment_budget = np.int(np.ceil(0.05 * self.env.L))
    elif env_name == 'Ebola':
        treatment_budget = np.int(np.ceil(0.15 * self.env.L))
    self.policy_arguments = {'classifier': SKLogit2, 'regressor': Ridge, 'env': self.env,
                              'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                              'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                              'divide_evenly': False, 'argmaxer': self.argmaxer, 'q_model': None,
                              'bootstrap': bootstrap, 'initial_policy_parameter': None, 'q_fn': None,
                              'quantile': error_quantile}

    # Get settings dict for log
    self.settings = {'classifier': self.policy_arguments['classifier'].__name__,
                     'regressor': self.policy_arguments['regressor'].__name__,
                     'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                     'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                     'divide_evenly': self.policy_arguments['divide_evenly'], 'argmaxer': argmaxer_name}
    self.settings.update({'env_name': env_name, 'L': self.env.L, 'policy_name': policy_name,
                          'argmaxer_name': argmaxer_name, 'time_horizon': self.time_horizon,
                          'number_of_replicates': self.number_of_replicates})

    # Get filename base for saving results
    to_join = [env_name, policy_name, argmaxer_name, str(self.env.L), network_name,
               'sampling-dbn-run={}'.format(sampling_dbn_run)]
    if 'epsilon' in env_kwargs.keys():
      to_join.append(str(env_kwargs['epsilon']))
    self.basename = '_'.join(to_join)

  def run_for_sampling_dbn(self):
    # Multiprocess simulation replicates
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 48))
    pool = mp.Pool(processes=num_processes)
    if self.ignore_errors:
      results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
    else:
      results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])

    # Save results
    results_dict = {}
    q_fn_params_list = []
    bootstrap_dbns = []
    for d in results_list:
      if d is not None:
        for k, v in d.items():
          # results_dict[k] = v['q_fn_params']
          q_fn_params_list.append(v['q_fn_params'])
          bootstrap_dbns.append(v['q_fn_bootstrap_dbn'])

    # For each bootstrap distribution, do ks-test against observed dbn and get coverage
    # ToDo: using distribution of first parameter only
    q_fn_params_list = np.array(q_fn_params_list)
    true_q_fn_params = q_fn_params_list.mean(axis=0)  # Treating mean parameter values as truth to compute coverage
    num_params = q_fn_params_list.shape[1]
    bootstrap_pvals = []
    coverages = []
    for bootstrap_dbn in bootstrap_dbns:
      bootstrap_pvals_rep = []
      coverages_rep = []
      bootstrap_dbn = np.array(bootstrap_dbn)
      for param in range(num_params):
        # Do ks-test
        bootstrap_pvals_rep.append(float(ks_2samp(q_fn_params_list[:, param], bootstrap_dbn[:, param])[0]))
        # Get coverage
        conf_interval = np.percentile(bootstrap_dbn[:, param], [2.5, 97.5])
        if true_q_fn_params[param] >= conf_interval[0] and true_q_fn_params[param] <= conf_interval[1]:
          coverages_rep.append(1)
        else:
          coverages_rep.append(0)
      bootstrap_pvals.append(bootstrap_pvals_rep)
      coverages.append(coverages_rep)

    # Test for normality
    pvals = normaltest(np.array(q_fn_params_list)).pvalue
    pvals = [float(pval) for pval in pvals]
    coverages = np.array(coverages).mean(axis=0)
    coverages = [float(c) for c in coverages]
    results_dict['pvals'] = pvals
    results_dict['bootstrap_pvals'] = bootstrap_pvals
    results_dict['coverages'] = coverages
    self.save_results(results_dict)
    return

  def run(self):
    # Multiprocess simulation replicates
    # num_processes = int(np.min((self.number_of_replicates, mp.cpu_count() / 2)))
    np.random.seed(self.seed)
    num_processes = self.number_of_replicates
    pool = mp.Pool(processes=num_processes)
    # iterim_results_list = []
    # results_list = []
    if self.ignore_errors:
      results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
    else:
      # results_list = pool.map_async(self.episode, [i for i in range(self.number_of_replicates)])
      results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])
    # for rep in range(self.number_of_replicates):
    #   iterim_results_list.append(pool.apply_async(self.episode, args=(rep,)))
    # for res in iterim_results_list:
    #   try:
    #     results_list.append(res.get(timeout=1000))
    #   except mp.context.TimeoutError:
    #     pass
    # pool.close()
    # pool.join()

    # results_list = pool.map(self.episode, range(self.number_of_replicates))

    # Save results
    results_dict = {}
    for d in results_list:
      if d is not None:
        for k, v in d.items():
          results_dict[k] = v
    # results_dict = {k: v for d in results_list for k, v in d.items() if d is not None}
    list_of_scores = [v['score'] for v in results_dict.values()]
    list_of_mean_losses = [v['mean_losses'] for v in results_dict.values()]
    list_of_max_losses = [v['max_losses'] for v in results_dict.values()]

    if list_of_mean_losses[0][0] is not None:
      mean_mean_losses = np.array(list_of_mean_losses).mean(axis=0)
      mean_max_losses = np.array(list_of_max_losses).mean(axis=0)
    else:
      mean_mean_losses = None
      mean_max_losses = None

    mean, se = float(np.mean(list_of_scores)), float(np.std(list_of_scores) / np.sqrt(len(list_of_scores)))
    results_dict['mean'] = mean
    results_dict['se'] = se
    results_dict['mean_mean_losses'] = mean_mean_losses
    results_dict['mean_max_losses'] = mean_max_losses
    self.save_results(results_dict)
    print('mean: {} se: {}'.format(mean, se))
    return

  def episode_wrapper(self, replicate):
    try:
      return self.episode(replicate)
    except:
      return None

  def episode(self, replicate):
    np.random.seed(int(self.seed*self.number_of_replicates + replicate))
    episode_results = {'score': None, 'runtime': None}
    t0 = time.time()
    self.env.reset()

    # Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    mean_losses = []
    max_losses = []
    for t in range(self.time_horizon-2):
      a, info = self.policy(**self.policy_arguments)
      self.policy_arguments['planning_depth'] = self.time_horizon - t

      # For pre-fit q_fn
      if info is not None:
        if 'q_fn' in info.keys():
          self.policy_arguments['q_fn'] = info['q_fn']
        if 'mean_loss' in info.keys():
          mean_loss = float(info['mean_loss'])
          max_loss = float(info['max_loss'])
        else:
          mean_loss = None
          max_loss = None
      else:
        mean_loss = None
        max_loss = None
      mean_losses.append(mean_loss)
      max_losses.append(max_loss)

      # For policy search
      # if 'initial_policy_parameter' in info.keys():
      #   self.policy_arguments['initial_policy_parameter'] = info['initial_policy_parameter']

      self.env.step(a)
      print('{} info {}'.format(t, info))
    t1 = time.time()
    # score = np.mean(self.env.Y)
    score = np.mean(self.env.current_infected)
    episode_results['score'] = float(score)
    episode_results['runtime'] = float(t1 - t0)
    episode_results['mean_losses'] = mean_losses
    episode_results['max_losses'] = max_losses

    if self.fit_qfn_at_end:
      NUM_BOOTSTRAP_SAMPLES = self.number_of_replicates

      # Get q-function parameters
      q_fn_policy_params = copy.deepcopy(self.policy_arguments)
      q_fn_policy_params['classifier'] = SKLogit2
      q_fn_policy_params['regressor'] = Ridge
      q_fn_policy = policy_factory(self.sampling_dbn_estimator)
      _, q_fn_policy_info = q_fn_policy(**q_fn_policy_params)

      # Get bootstrap dbn of q-function parameters
      bootstrap_dbn = []
      q_fn_policy_params['bootstrap'] = True
      for sample in range(NUM_BOOTSTRAP_SAMPLES):
        _, bootstrap_q_fn_policy_info = q_fn_policy(**q_fn_policy_params)
        bootstrap_dbn.append([float(t) for t in bootstrap_q_fn_policy_info['q_fn_params']])

      episode_results['q_fn_params'] = [float(t) for t in q_fn_policy_info['q_fn_params']]
      episode_results['q_fn_bootstrap_dbn'] = bootstrap_dbn

    # print(np.mean(self.env.Y[-1,:]))
    print(episode_results)
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
      self.policy_arguments['planning_depth'] = self.time_horizon - 2 - t
      self.env.step(a)
      if t in times_to_evaluate:
        mb_be, mf_be = self.generate_bootstrap_distributions(num_bootstrap_samples)
        bootstrap_results['mb_be'].append(mb_be)
        bootstrap_results['mf_be'].append(mf_be)
    return bootstrap_results

  def compare_probability_estimates_episode(self, model_constructor_list):
    """
    Assuming sis generative model.
    :param model_constructor_list: e.g. [KerasLogit, SKLogit, ...]
    :return:
    """
    # Initialize results dict
    estimates_results = {'mb_loss': []}
    for model in model_constructor_list:
      estimates_results.update({'obs_data_loss_{}'.format(model.__name__): [],
                                'sim_data_loss_{}'.format(model.__name__): []})

    self.env.reset()
    # Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])

    for t in range(self.time_horizon - 2):
      a, _ = self.random_policy(**self.policy_arguments)
      self.env.step(a)

      target = np.hstack(self.env.y).astype(float)
      features = np.vstack(self.env.X)
      p = np.hstack(self.env.true_infection_probs)

      # Fit models to observations
      for model in model_constructor_list:
        m = model()
        m.fit(features, target, weights=None, exclude_neighbor_features=self.env.add_neighbor_sums)
        phat = m.predict_proba(features)[:,-1]
        # loss = np.mean(self.cross_entropy(phat, p))
        loss = np.mean((phat - p)**2)
        estimates_results['obs_data_loss_{}'.format(model.__name__)].append(float(loss))

      # Fit sis model
      eta = fit_sis_transition_model(self.env)
      simulation_env = simulate_from_SIS(self.env, eta, 5,self.settings['treatment_budget'], n_rep=5)
      sis_losses = []
      for x, t in zip(self.env.X_raw, range(len(self.env.X_raw))):
        s, a, y = x[:,0], x[:,1], x[:,2]
        phat = simulation_env.infection_probability(a, y, s)
        p_t = self.env.true_infection_probs[t]
        # sis_losses.append(self.cross_entropy(phat, p_t))
        loss = np.mean((phat - p_t)**2)
        sis_losses.append(loss)
      estimates_results['mb_loss'].append(float(np.mean(sis_losses)))

      # Fit model to simulation data
      sim_target = np.hstack(simulation_env.y).astype(float)
      sim_features = np.vstack(simulation_env.X)
      for model in model_constructor_list:
        m = model()
        m.fit(sim_features, sim_target, weights=None, exclude_neighbor_features=simulation_env.add_neighbor_sums)
        phat = m.predict_proba(features)[:,-1]
        # loss = np.mean(self.cross_entropy(phat, p))
        loss = np.mean((phat - p)**2)
        estimates_results['sim_data_loss_{}'.format(model.__name__)].append(float(loss))

      K.clear_session()
    return estimates_results

  def probability_episode_wrapper(self, replicate):
    np.random.seed(replicate)
    results = self.compare_probability_estimates_episode(self.model_constructor_list)
    return {replicate: results}

  def run_compare_probability_estimates(self, model_constructor_list=[SKLogit, SKLogit2]):
    self.basename = '_'.join(['prob_estimates', self.basename])
    self.model_constructor_list = model_constructor_list

    num_processes = int(np.min((self.number_of_replicates, mp.cpu_count() / 2)))
    pool = mp.Pool(processes=num_processes)
    for rep in range(self.number_of_replicates):
     pool.apply_async(self.probability_episode_wrapper, args=(rep,))
    results_list = results_.get(timeout=240)

    # Save results
    results_dict = {k: v for d in results_list for k, v in d.items()}
    self.save_results(results_dict)

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
    # results_list = pool.map(self.bootstrap_distribution_episode_wrapper, range(self.number_of_replicates))
    results1 = pool.apply_async(self.bootstrap_distribution_episode_wrapper, args=range(self.number_of_replicates))
    results_list = results1.get(timeout=240)

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
