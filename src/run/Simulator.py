# -*- coding: utf-8 -*-N
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

from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.sis.simulate_from_sis import simulate_from_SIS
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory
from src.estimation.stacking.bellman_error_bootstrappers import bootstrap_rollout_qfn, bootstrap_SIS_mb_qfn
from scipy.spatial.distance import cdist
from scipy.stats import normaltest, ks_2samp
from scipy.linalg import sqrtm as sqrtm
import itertools
import copy

from src.estimation.q_functions.model_fitters import KerasRegressor, SKLogit, SKLogit2
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

# import keras.backend as K
def bootstrap_coverages(bootstrap_dbns_, q_fn_params_list_):
  q_fn_params_list_ = np.array(q_fn_params_list_)
  true_q_fn_params_ = q_fn_params_list_.mean(axis=0)
  q_fn_params_list_ = q_fn_params_list_ - true_q_fn_params_
  num_params = len(true_q_fn_params_)
  bootstrap_pvals = []
  coverages = []
  for it, bootstrap_dbn in enumerate(bootstrap_dbns_):
    bootstrap_pvals_rep = []
    coverages_rep = []
    bootstrap_dbn = np.array(bootstrap_dbn)
    for param in range(num_params):
      # Do ks-test
      bootstrap_pvals_rep.append(float(ks_2samp(q_fn_params_list_[:, param], bootstrap_dbn[:, param])[1]))
      # Get coverage
      conf_interval = np.percentile(bootstrap_dbn[:, param], [2.5, 97.5])
      if true_q_fn_params_[param] >= conf_interval[0] and true_q_fn_params_[param] <= conf_interval[1]:
      # if 0 >= conf_interval[0] and 0 <= conf_interval[1]:
        coverages_rep.append(1)
      else:
        coverages_rep.append(0)
    coverages.append(coverages_rep)
    bootstrap_pvals.append(bootstrap_pvals_rep)
  return bootstrap_pvals, np.array(coverages).mean(axis=0)


class Simulator(object):
  def __init__(self, lookahead_depth, env_name, time_horizon, number_of_replicates, policy_name, argmaxer_name, gamma,
               evaluation_budget, env_kwargs, network_name, bootstrap, seed, error_quantile,
               sampling_dbn_run=False, sampling_dbn_estimator=None, fit_qfn_at_end=False, variance_only=False,
               parametric_bootstrap=False, ignore_errors=False, fname_addendum=None, save_features=False):
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
    self.variance_only = variance_only
    self.parametric_bootstrap = parametric_bootstrap
    self.save_features = save_features

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
                     'divide_evenly': self.policy_arguments['divide_evenly'], 'argmaxer': argmaxer_name,
                     'evaluation_policy': self.sampling_dbn_estimator}
    self.settings.update({'env_name': env_name, 'L': self.env.L, 'policy_name': policy_name,
                          'argmaxer_name': argmaxer_name, 'time_horizon': self.time_horizon,
                          'number_of_replicates': self.number_of_replicates})

    # Get filename base for saving results
    to_join = [env_name, policy_name, argmaxer_name, str(self.env.L), network_name,
               'eval-policy={}'.format(self.sampling_dbn_estimator)]
    if sampling_dbn_run:
      to_join.append('eval={}'.format(self.sampling_dbn_estimator))
    if 'epsilon' in env_kwargs.keys():
      to_join.append(str(env_kwargs['epsilon']))
    if fname_addendum is not None:
      to_join.append(fname_addendum)
    self.basename = '_'.join(to_join)

  def run_for_bootstrap(self):
    if self.parametric_bootstrap:
      self.run_for_parametric_bootstrap()
    else:
      if self.variance_only:
        self.run_for_nonparametric_boot_variance()
      else:
        self.run_for_nonparametric_boot_sampling_dbn()

  def run_for_parametric_bootstrap(self):
    # ToDo: Assuming model is SIS!

    # Multiprocess simulation replicates
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 48))
    pool = mp.Pool(processes=num_processes)
    if self.number_of_replicates > 1:
      if self.ignore_errors:
        results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
      else:
        results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])
    else:
      results_list = [self.episode(0)]

    # Save results
    results_dict = {}
    q_fn_params_list = []
    q_fn_params_raw_list = []
    bootstrap_dbns = []
    raw_bootstrap_dbns = []

    for ix, d_ in enumerate(results_list):
      q_fn_params_list.append(d_[ix]['q_fn_params'])
      q_fn_params_raw_list.append(d_[ix]['q_fn_params_raw'])
      bootstrap_dbns.append(d_[ix]['q_fn_bootstrap_dbn'])
      raw_bootstrap_dbns.append(d_[ix]['q_fn_bootstrap_dbn_raw'])

    # For each bootstrap distribution, do ks-test against observed dbn and get coverage
    q_fn_params_list = np.array(q_fn_params_list)
    true_q_fn_params = q_fn_params_list.mean(
      axis=0)  # Treating mean parameter values as truth to compute coverage
    biases = np.array([p - true_q_fn_params for p in q_fn_params_list])
    bias = biases.mean(axis=0)
 
    bootstrap_pvals, coverages = bootstrap_coverages(bootstrap_dbns, q_fn_params_list)
    raw_bootstrap_pvals, raw_coverages = bootstrap_coverages(raw_bootstrap_dbns, q_fn_params_raw_list)
    
    # Test for normality
    pvals = normaltest(np.array(q_fn_params_list)).pvalue
    pvals = [float(pval) for pval in pvals]
    beta_vars = np.var(np.array(q_fn_params_list), axis=0)
    beta_vars = [float(b) for b in beta_vars]
    results_dict['pvals'] = pvals
    results_dict['beta_vars'] = beta_vars
    results_dict['bias'] = [float(b) for b in bias]
    results_dict['coverages'] = [float(c) for c in raw_coverages]
    results_dict['bootstrap_pvals'] = bootstrap_pvals
    self.save_results(results_dict)

  def run_for_nonparametric_boot_variance(self):
    # Multiprocess simulation replicates
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 48))
    pool = mp.Pool(processes=num_processes)
    if self.number_of_replicates > 1:
      if self.ignore_errors:
        results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
      else:
        results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])
    else:
      results_list = [self.episode(0)]

    # Save results
    results_dict = {}
    q_fn_params_list = []
    q_fn_params_raw_list = []
    counts = []
    eigs_list = []
    acfs_list = []
    ys_list = []
    zbar_list = []
    zvar_list = []
    zvar_naive_list = []
    for d in results_list:
      if d is not None:
        for k, v in d.items():
          # results_dict[k] = v['q_fn_params']
          q_fn_params_list.append(v['q_fn_params'])
          q_fn_params_raw_list.append(v['q_fn_params_raw'])
          # counts.append(v['nonzero_counts'])
          eigs_list.append(v['eigs'])
          acfs_list.append(v['acfs'])
          ys_list.append(v['ys'])
          zbar_list.append(v['zbar'])
          zvar_list.append(v['zvar'])
          zvar_naive_list.append(v['zvar_naive'])

    # mean_counts = np.array(counts).mean(axis=0)
    # mean_counts = [float(m) for m in mean_counts]
    acfs = [float(acf) for acf in np.array(acfs_list).mean(axis=0)]

    # For each bootstrap distribution, do ks-test against observed dbn and get coverage
    q_fn_params_list = np.array(q_fn_params_list)
    true_q_fn_params = q_fn_params_list.mean(axis=0)  # Treating mean parameter values as truth to compute coverage
    num_params = q_fn_params_list.shape[1]
    biases = np.array([p - true_q_fn_params for p in q_fn_params_list])
    bias = biases.mean(axis=0)

    # Get 'true' and estimated covariance matrices
    zvars = np.array(zvar_list)
    zbars = []
    # autocovariance = np.zeros((int(np.max(self.env.pairwise_distances)), 4, 4))  # For checking assumption 2.1.a in dependent wild bootstrap paper
    max_dist = int(np.max(self.env.pairwise_distances))
    autocov_dict = {k_: {(l_,  t_): np.zeros((4, 4))
                         for l_, t_ in itertools.product(range(self.env.L), range(self.time_horizon+1))
                         if t_ + self.env.pairwise_distances[0, l_] == k_} for k_ in range(self.time_horizon + max_dist+1)}
    true_q_fn_params_raw = np.array(q_fn_params_raw_list).mean(axis=0)
    sq_residual_means = np.zeros(zbar_list[0][0].shape[0])
    sq_residual_vars = np.zeros_like(sq_residual_means) 

    for X_raw, y in zbar_list:
      # Compute variances
      y_hat = np.dot(X_raw, true_q_fn_params_raw)
      error = y - y_hat
      
      x_raw_e_0 = X_raw[0] * error[0]
      x_raw_0_e_sq = np.dot(x_raw_e_0, np.ones_like(x_raw_e_0))**2 # Wald device

      # Compute autocovariances and means of squared residuals 
      for i, x_raw in enumerate(X_raw):
        x_raw_e = x_raw * error[i]
        x_raw_e_sq = np.dot(x_raw_e, np.ones_like(x_raw_e))**2 # Wald device 

        l = i % self.env.L
        t = i // self.env.L
        d = (l, t)
        k = int(self.env.pairwise_distances[0, l] + t)
        autocov_dict[k][d] += np.outer(x_raw_e_sq, x_raw_0_e_sq) / (self.time_horizon* len(zbar_list))
        sq_residual_means[i] += x_raw_e_sq / len(zbar_list)         
 
    # Second pass for autocovariancs of squared residuals
    autocov_sq_dict = {k_: {(l_, t_): [] 
                               for l_, t_ in itertools.product(range(self.env.L), range(self.time_horizon))
                               if t_ + self.env.pairwise_distances[0, l_] == k_} for k_ in range(self.time_horizon + max_dist + 1)}
    zbar_autocovs = []
    for X_raw, y in zbar_list:
      y_hat = np.dot(X_raw, true_q_fn_params_raw)
      error = y - y_hat
      # zbars.append(np.dot(np.ones(X_raw.shape[1]), np.multiply(X_raw.T, error)).sum() / X_raw.shape[0])
      X_raw_dot_y = np.dot(np.ones(X_raw.shape[1]), np.multiply(X_raw.T, error))
      zbars.append(X_raw_dot_y.sum() / np.sqrt(X_raw.shape[0]))
      zbar_autocovs.append([X_raw_dot_y[0]*x for x in X_raw_dot_y]) 
      
      x_raw_e_0 = X_raw[0] * error[0] 
      x_raw_e_0_sq = np.dot(x_raw_e_0, np.ones_like(x_raw_e_0))**2 # Wald device
      x_raw_e_0_sq_mean = sq_residual_means[0] 

      # Compute autocovariances and autocovariances of squared residuals 
      for i, x_raw in enumerate(X_raw):
        x_raw_e = x_raw * error[i]
        x_raw_e_sq = np.dot(x_raw_e, np.ones_like(x_raw_e))**2 # Wald device 
        l = i % self.env.L
        t = i // self.env.L
        d = (l,  t)
        k = int(self.env.pairwise_distances[0, l]) + t
        x_raw_e_sq_mean = sq_residual_means[i]
        sq_residual_vars[i] += (x_raw_e_sq - x_raw_e_sq_mean)**2 / len(zbar_list)
        cross_product = (x_raw_e_sq - x_raw_e_sq_mean) * (x_raw_e_0_sq - x_raw_e_0_sq_mean)
        autocov_sq_dict[k][d].append(cross_product)

    # Get autocov sum
    autocovs = []
    autocovs_sq_max = []
    autocovs_sq_mean = []
    for k in autocov_dict.keys():
      eigs_k = [np.real(np.linalg.eig(C)[0]).max() for C in autocov_dict[k].values()]
      autocovs.append(np.max(eigs_k))
      autocovs_sq_k = [np.abs(np.mean(C)) for C in autocov_sq_dict[k].values() if C]
      if autocovs_sq_k:
        autocovs_sq_max.append(np.max(autocovs_sq_k))
        autocovs_sq_mean.append(np.mean(autocovs_sq_k))

    autocovs_sq_max = np.round(autocovs_sq_max, 2)
    autocovs_sq_mean = np.round(autocovs_sq_mean, 2)
    autocovs_cum = np.cumsum(autocovs)
    autocovs_sq_cum = np.cumsum(autocovs_sq_max)

    zbars = np.array(zbars)
    true_cov = np.var(zbars.T)
    est_cov = np.mean(zvars)
    est_cov_naive = np.mean(np.array(zvar_naive_list))

    pdb.set_trace()

    # Test for normality
    pvals = normaltest(np.array(q_fn_params_list)).pvalue
    raw_pvals = normaltest(np.array(q_fn_params_raw_list)).pvalue

    pvals = [float(pval) for pval in pvals]
    eig_vars = np.var(eigs_list, axis=0)
    eig_vars = [float(v) for v in eig_vars]
    beta_vars = np.var(np.array(q_fn_params_list), axis=0)
    beta_vars = [float(b) for b in beta_vars]
    results_dict['pvals'] = pvals
    results_dict['eig_vars'] = eig_vars
    results_dict['beta_vars'] = beta_vars
    # results_dict['mean_counts'] = mean_counts
    results_dict['bias'] = [float(b) for b in bias]
    results_dict['betas'] = [float(b) for b in true_q_fn_params]
    results_dict['acfs'] = [float(a) for a in acfs]
    results_dict['autocovs_sq'] = [float(a) for a in autocovs_sq_max]
    self.save_results(results_dict)

  def run_for_nonparametric_boot_sampling_dbn(self):
    # Multiprocess simulation replicates
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 48))
    pool = mp.Pool(processes=num_processes)
    if self.number_of_replicates > 1:
      if self.ignore_errors:
        results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
      else:
        results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])
    else:
      results_list = [self.episode(0)]

    # Save results
    results_dict = {}
    q_fn_params_list = []
    q_fn_params_raw_list = []
    bootstrap_dbns = []
    raw_bootstrap_dbns = []
    counts = []
    eigs_list = []
    acfs_list = []
    ys_list = []
    zbar_list = []
    zvar_list = []
    for d in results_list:
      if d is not None:
        for k, v in d.items():
          # results_dict[k] = v['q_fn_params']
          q_fn_params_list.append(v['q_fn_params'])
          q_fn_params_raw_list.append(v['q_fn_params_raw'])
          bootstrap_dbns.append(v['q_fn_bootstrap_dbn'])
          raw_bootstrap_dbns.append(v['q_fn_bootstrap_dbn_raw'])
          counts.append(v['nonzero_counts'])
          eigs_list.append(v['eigs'])
          acfs_list.append(v['acfs'])
          ys_list.append(v['ys'])
          zbar_list.append(v['zbar'])
          zvar_list.append(v['zvar'])
    mean_counts = np.array(counts).mean(axis=0)
    mean_counts = [float(m) for m in mean_counts]
    acfs = [float(acf) for acf in np.array(acfs_list).mean(axis=0)]

    # For each bootstrap distribution, do ks-test against observed dbn and get coverage
    # ToDo: using distribution of first parameter only
    q_fn_params_list = np.array(q_fn_params_list)
    true_q_fn_params = q_fn_params_list.mean(axis=0)  # Treating mean parameter values as truth to compute coverage
    num_params = q_fn_params_list.shape[1]
    biases = np.array([p - true_q_fn_params for p in q_fn_params_list])
    bias = biases.mean(axis=0) 
    
    bootstrap_pvals, coverages = bootstrap_coverages(bootstrap_dbns, q_fn_params_list)
    raw_bootstrap_pvals, raw_coverages = bootstrap_coverages(raw_bootstrap_dbns, q_fn_params_raw_list)

    pdb.set_trace()
    # Test for normality
    pvals = normaltest(np.array(q_fn_params_list)).pvalue
    pvals = [float(pval) for pval in pvals]
    beta_vars = np.var(np.array(q_fn_params_list), axis=0)
    beta_vars = [float(b) for b in beta_vars]
    coverages = [float(c) for c in coverages]
    results_dict['pvals'] = pvals
    results_dict['bootstrap_pvals'] = raw_bootstrap_pvals
    results_dict['coverages'] = [float(c) for c in raw_coverages]
    results_dict['beta_vars'] = beta_vars 
    results_dict['mean_counts'] = mean_counts
    results_dict['bias'] = [float(b) for b in bias]
    results_dict['betas'] = [float(b) for b in true_q_fn_params]
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
    # results_dict = {}
    # for d in results_list:
    #   if d is not None:
    #     for k, v in d.items():
    #       results_dict[k] = v
    results_dict = {k: v for d in results_list for k, v in d.items() if d is not None}
    # list_of_scores = [v['score'] for v in results_dict.values()]
    # list_of_mean_losses = [v['mean_losses'] for v in results_dict.values()]
    # list_of_max_losses = [v['max_losses'] for v in results_dict.values()]

    # if list_of_mean_losses[0][0] is not None:
    #   mean_mean_losses = np.array(list_of_mean_losses).mean(axis=0)
    #   mean_max_losses = np.array(list_of_max_losses).mean(axis=0)
    # else:
    #   mean_mean_losses = None
    #   mean_max_losses = None

    # mean, se = float(np.mean(list_of_scores)), float(np.std(list_of_scores) / np.sqrt(len(list_of_scores)))
    # results_dict['mean'] = mean
    # results_dict['se'] = se
    # results_dict['mean_mean_losses'] = mean_mean_losses
    # results_dict['mean_max_losses'] = mean_max_losses
    self.save_results(results_dict)
    return

  def episode_wrapper(self, replicate):
    return self.episode(replicate)

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
      print(t)
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
    t1 = time.time()
    # score = np.mean(self.env.Y)
    score = np.mean(self.env.current_infected)
    episode_results['score'] = float(score)
    episode_results['runtime'] = float(t1 - t0)
    episode_results['mean_losses'] = mean_losses
    episode_results['max_losses'] = max_losses

    if self.fit_qfn_at_end:
      # Get q-function parameters
      q_fn_policy_params = copy.deepcopy(self.policy_arguments)
      q_fn_policy_params['rollout'] = False
      q_fn_policy_params['rollout_env'] = None
      q_fn_policy_params['rollout_policy'] = None
      q_fn_policy_params['time_horizon'] = self.time_horizon
      q_fn_policy_params['classifier'] = SKLogit2
      q_fn_policy_params['regressor'] = Ridge
      q_fn_policy_params['bootstrap'] = False
      q_fn_policy = policy_factory(self.sampling_dbn_estimator)
      _, q_fn_policy_info = q_fn_policy(**q_fn_policy_params)
      episode_results['q_fn_params'] = [float(t) for t in q_fn_policy_info['q_fn_params']]
      episode_results['q_fn_params_raw'] = [float(t) for t in q_fn_policy_info['q_fn_params_raw']]

      if self.parametric_bootstrap:
        pass
      #   NUM_BOOTSTRAP_SAMPLES = self.number_of_replicates

      #   # Fit transition model and get rollout_env for parametric boot
      #   eta_hat, _ = fit_infection_prob_model(self.env, None)
      #   eta_hat_cov = self.env.mb_covariance(eta_hat)
      #   rollout_env_kwargs = {'L': self.env.L, 'omega': self.env.omega, 'generate_network': self.env.generate_network,
      #                         'initial_infections': self.env.initial_infections,
      #                         'add_neighbor_sums': self.env.add_neighbor_sums, 'epsilon': 0.0,
      #                         'compute_pairwise_distances': self.env.compute_pairwise_distances,
      #                         'dummy': self.env.dummy, 'eta': eta_hat}
      #   rollout_env = environment_factory('sis', **rollout_env_kwargs)
      #   q_fn_policy_params['rollout'] = True
      #   q_fn_policy_params['rollout_env'] = rollout_env
      #   q_fn_policy_params['rollout_policy'] = None

      #   # Get bootstrap dbn of q-function parameters
      #   bootstrap_dbn = []
      #   raw_bootstrap_dbn = []
      #   q_fn_policy_params['bootstrap'] = True

      #   for sample in range(NUM_BOOTSTRAP_SAMPLES):
      #     _, bootstrap_q_fn_policy_info = q_fn_policy(**q_fn_policy_params)
      #     bootstrap_dbn.append([float(t) for t in bootstrap_q_fn_policy_info['q_fn_params']])
      #     raw_bootstrap_dbn.append([float(t) for t in bootstrap_q_fn_policy_info['q_fn_params_raw']])

      #   # bootstrap_dbn = np.array(bootstrap_dbn) - np.array(q_fn_policy_info['q_fn_params'])
      #   # raw_bootstrap_dbn = np.array(raw_bootstrap_dbn) - np.array(q_fn_policy_info['q_fn_params_raw'])
      #   episode_results['q_fn_bootstrap_dbn'] = bootstrap_dbn
      #   episode_results['q_fn_bootstrap_dbn_raw'] = raw_bootstrap_dbn
      else:
        # episode_results['nonzero_counts'] = q_fn_policy_info['nonzero_counts']
        episode_results['eigs'] = q_fn_policy_info['eigs']
        episode_results['acfs'] = q_fn_policy_info['acfs']
        episode_results['ys'] = q_fn_policy_info['ys']
        episode_results['zbar'] = q_fn_policy_info['zbar']
        episode_results['zvar'] = q_fn_policy_info['zvar']
        episode_results['zvar_naive'] = q_fn_policy_info['zvar_naive']

      #   if not self.variance_only:
      #     NUM_BOOTSTRAP_SAMPLES = self.number_of_replicates
      #     # Get bootstrap dbn of q-function parameters
      #     bootstrap_dbn = []
      #     raw_bootstrap_dbn = []
      #     q_fn_policy_params['bootstrap'] = True

      #     if 'wild' in self.sampling_dbn_estimator: # ToDo: encapsulate in separate function
      #       N = len(self.env.X)*self.env.L
      #       BANDWIDTH = 0.1
      #
      #       # Construct pairwise distances matrices
      #       pairwise_t = cdist(np.arange(self.env.T).reshape(-1, 1), np.arange(self.env.T).reshape(-1, 1))
      #       pairwise_t /= (np.max(pairwise_t) / BANDWIDTH)
      #       pairwise_l = self.env.pairwise_distances
      #       pairwise_l /= (np.max(pairwise_l) / BANDWIDTH)
      #
      #       # Construct kernels
      #       K_l = np.exp(-np.multiply(pairwise_l, pairwise_l)*100)
      #       K_t = np.exp(-np.multiply(pairwise_t, pairwise_t)*100)
      #       K = np.kron(K_t, K_l)
      #       q_fn_policy_params['cov_sqrt'] = np.real(sqrtm(K))
      #
      #     for sample in range(NUM_BOOTSTRAP_SAMPLES):
      #       _, bootstrap_q_fn_policy_info = q_fn_policy(**q_fn_policy_params)
      #       bootstrap_dbn.append([float(t) for t in bootstrap_q_fn_policy_info['q_fn_params']])
      #       raw_bootstrap_dbn.append([float(t) for t in bootstrap_q_fn_policy_info['q_fn_params_raw']])
      #     episode_results['q_fn_bootstrap_dbn'] = bootstrap_dbn
      #     episode_results['q_fn_bootstrap_dbn_raw'] = raw_bootstrap_dbn

      #   print('GOT HERE')

    if self.save_features and self.number_of_replicates == 1:  # Save raw features, responses, adjacency matrix
      X_raw = self.env.X_raw
      y = self.env.y
      adjacency_mat = self.env.adjacency_matrix
      save_dict = {'X_raw': X_raw, 'y': y, 'adjacency_mat': adjacency_mat}
      prefix = os.path.join(pkg_dir, 'analysis', 'observations', self.basename)
      suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      filename = '{}_{}.npy'.format(prefix, suffix)
      np.save(filename, save_dict)

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
