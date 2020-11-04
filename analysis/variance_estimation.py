import argparse
import numpy as np
import scipy.linalg as la
from scipy.stats import normaltest
import pdb
import multiprocessing as mp
from functools import partial
import yaml
import datetime
import matplotlib.pyplot as plt
from numba import njit

import os
this_dir = os.path.dirname(os.path.abspath(__file__))


@njit
def get_pairwise_distances(grid_size):
  length = int(np.floor(np.sqrt(grid_size)))
  true_grid_size = length ** 2
  pairwise_distances = np.zeros((true_grid_size, true_grid_size))
  for i in range(length):
    for j in range(length):
      for iprime in range(length):
        for jprime in range(length):
          index_1 = length * i + j
          index_2 = length * iprime + jprime
          distance = np.abs(i - iprime) + np.abs(j - jprime)
          pairwise_distances[index_1, index_2] = distance
  return pairwise_distances


@njit
def get_exponential_gaussian_covariance_helper(vars, beta1=1, beta2=2, grid_size=100):
  """
  Using gen model from "Optimal block size for variance estimation by a spatial block bootstrap method".

  :param grid_size: Will sample observations on floor(sqrt(grid_size)) x floor(sqrt(grid_size)) lattice
  """
  # Get covariance matrix
  length = int(np.floor(np.sqrt(grid_size)))
  true_grid_size = length**2
  cov = np.zeros((true_grid_size, true_grid_size))
  for i in range(length):
    for j in range(length):
      for iprime in range(length):
        for jprime in range(length):
          index_1 = length*i + j
          index_2 = length*iprime + jprime
          weighted_distance = beta1*np.abs(i - iprime) + beta2*np.abs(j - jprime)
          cov_ = np.exp(-weighted_distance)
          if i == j:
            cov_ *= vars[i]
          cov[index_1, index_2] = cov_

  return cov


def get_exponential_gaussian_covariance(beta1=1, beta2=1, grid_size=100, heteroskedastic=False):
  if heteroskedastic:
    vars = np.linspace(2/grid_size, 2, grid_size)
  else:
    vars = np.ones(grid_size)
  cov = get_exponential_gaussian_covariance_helper(beta1=beta1, beta2=beta2, grid_size=grid_size, vars=vars)
  root_cov = np.real(la.sqrtm(cov))
  return cov, root_cov


def generate_gaussian(root_cov):
  L = root_cov.shape[0]
  y_raw = np.random.normal(size=L)
  y = np.dot(root_cov, y_raw)
  return y


def get_autocov_sq_sequence(observations, pairwise_distances):
  """
  Get sequence of max autocovariance between squared residuals at location 0 and locations at each distance,
  as distance increases.
  """
  max_dist = np.max(pairwise_distances).astype(int)
  n_rep, L = observations.shape
  length = np.sqrt(L).astype(int)
  max_autocov_sq_sequence = np.zeros(max_dist+1)

  # Need mean squared residuals for computing covariances between squared residuals
  sample_means = observations.mean(axis=1)[:, None]
  squared_residuals = (observations - sample_means)**2
  mean_squared_residuals = squared_residuals.mean(axis=0)
  centered_squared_residuals = squared_residuals - mean_squared_residuals

  # Loop through distances and get the max covariance between location 0 and locations at that distance
  res_sq_0 = centered_squared_residuals[:, 0]
  for i in range(length):
    for j in range(length):
      d = i + j
      ix = length*i + j
      res_sq_ix = centered_squared_residuals[:, ix]
      autocov = np.abs(np.dot(res_sq_0, res_sq_ix) / n_rep)
      if autocov > max_autocov_sq_sequence[d]:
        max_autocov_sq_sequence[d] = autocov

  return max_autocov_sq_sequence


def get_sigma_sq_infty(observations):
  """
  sigma_sq_infty as defined in DWB paper.
  """
  num_rep, L = observations.shape
  x_bar = observations.mean(axis=0)
  x_0 = observations[:, 0] - x_bar[0]
  x_0_sq = x_0**2
  x_0_sq_mean = np.mean(x_0_sq)
  x_0_sq_centered = x_0_sq - x_0_sq_mean
  sigma_sq_infty = 0.
  sq_residual_cov = 0.

  for l in range(L):
    x_l = observations[:, l] - x_bar[l]
    x_l_sq = x_l**2
    x_l_sq_mean = np.mean(x_l_sq)
    x_l_sq_centered = x_l_sq - x_l_sq_mean
    cov_0l = np.dot(x_0, x_l) / num_rep
    cov_sq_0l = np.dot(x_0_sq_centered, x_l_sq_centered) / num_rep
    sigma_sq_infty += cov_0l
    sq_residual_cov += cov_sq_0l

  return sigma_sq_infty, sq_residual_cov


def draw_from_gaussian_and_estimate_var(ix, root_cov, kernel_weights, grid_size):
  y = generate_gaussian(root_cov)
  sigma_sq_hat = estimate_var(y, kernel_weights, grid_size)

  return y, sigma_sq_hat


@njit
def estimate_spatiotemporal_matrix_var(Y_centered, spatiotemporal_kernel_weights):
  N, p = Y_centered.shape

  cov_hat = np.zeros((p, p))
  for i in range(N):
    y_i = Y_centered[i]
    for j in range(N):
      y_j = Y_centered[j]
      kernel_weight = spatiotemporal_kernel_weights[i, j]
      for k_i in range(p):
        for k_j in range(p):
          cov_hat[k_i, k_j] += y_i[k_i]*y_j[k_j]*kernel_weight / N
  return cov_hat


@njit
def get_spatiotemporal_kernel(spatial_kernel_weights, temporal_kernel_weights, L, N):
  # if combination_function is None:
  #   combination_function = lambda x, y: np.min((x, y))
  spatiotemporal_kernel_weights = np.zeros((N, N))
  for i in range(N):
    t_i = i // L
    l_i = i % L
    for j in range(N):
      t_j = j // L
      l_j = j % L
      spatial_kernel = spatial_kernel_weights[l_i, l_j]
      temporal_kernel = temporal_kernel_weights[t_i, t_j]
      # kernel_weight = combination_function(spatial_kernel, temporal_kernel)
      kernel_weight = spatial_kernel + temporal_kernel
      spatiotemporal_kernel_weights[i, j] = kernel_weight
  return spatiotemporal_kernel_weights


@njit
def estimate_matrix_var(Y_centered, kernel_weights):
  # Estimate covariance from a list of mean-zero vectors Y_centered
  L, p = Y_centered.shape

  cov_hat = np.zeros((p, p))
  for i in range(L):
    for j in range(L):
      y_i = Y_centered[i]
      y_j = Y_centered[j]
      weighted_outer = np.outer(y_i, y_j)*kernel_weights[i, j]
      cov_hat += weighted_outer / L

  return cov_hat


def estimate_var(y, kernel_weights, grid_size):
  # Products of centered residuals
  y_centered = y - np.mean(y)
  cross_raw = np.outer(y_centered, y_centered)

  cross_kernel = np.multiply(cross_raw, kernel_weights)
  sigma_sq_hat = cross_kernel.sum() / grid_size

  return sigma_sq_hat


def construct_kernel_matrix_from_distances(kernel, pairwise_distances):
  grid_size = pairwise_distances.shape[0]
  kernel_weights = np.zeros((grid_size, grid_size))
  for i in range(grid_size):
    for j in range(grid_size):
      kernel_weights[i, j] = kernel(pairwise_distances[i, j])
  return kernel_weights


def get_var_estimate_mse(kernel, sigma_sq_infty, root_cov, pairwise_distances, n_rep=10000, pct_cores=0.25):
  var_estimates = np.zeros(0)
  grid_size = root_cov.shape[0]
  observations = np.zeros((0, grid_size))
  confidence_intervals = np.zeros((0, 2))

  kernel_weights = construct_kernel_matrix_from_distances(kernel, pairwise_distances)
  replicate = partial(draw_from_gaussian_and_estimate_var, root_cov=root_cov, kernel_weights=kernel_weights,
                      grid_size=grid_size)

  # Parallelize calculations on draws
  n_cpu = mp.cpu_count()
  n_processes = np.ceil(pct_cores*n_cpu).astype(int)
  pool = mp.Pool(processes=n_processes)
  res_list = pool.map(replicate, range(n_rep))

  # Collect results
  for y, sigma_sq_hat in res_list:
    observations = np.vstack((observations, y))
    var_estimates = np.hstack((var_estimates, sigma_sq_hat))

    # Compute CI
    y_bar = np.mean(y, axis=0)
    ci_upper = y_bar + 1.96*sigma_sq_hat / np.sqrt(grid_size)
    ci_lower = y_bar - 1.96*sigma_sq_hat / np.sqrt(grid_size)
    ci = np.column_stack((ci_lower, ci_upper))
    confidence_intervals = np.vstack((ci, confidence_intervals))

  # Compute coverage
  upper_coverage = confidence_intervals[:, 1] > 0
  lower_coverage = confidence_intervals[:, 0] < 0
  coverage = upper_coverage * lower_coverage
  mean_coverage = np.mean(coverage)

  mse_kernel = np.mean((var_estimates/sigma_sq_infty - 1)**2)
  # print('sigma sq infty: {}'.format(sigma_sq_infty))
  print('mse: {} coverage: {}'.format(mse_kernel, mean_coverage))
  return mse_kernel, mean_coverage


def bartlett(x, bandwidth):
  x_abs = np.abs(x) / bandwidth
  k = (1 - x_abs) * (x_abs <= 1)
  return k


def constant(x):
  return 1


def block(x, bandwidth):
  return np.abs(x) / bandwidth <= 1


def var_estimates(cov_name='exponential', n_bandwidths=10, betas=(0.1,), grid_size=100, n_rep=1000, pct_cores=0.25):
  """
  :param n_bandwidths: Bandwidths will be n_bandwidths values log-spaced between 1 and max(pairwise_distances).
  """

  pairwise_distances = get_pairwise_distances(grid_size)
  bandwidths = np.logspace(np.log10(1.), np.log10(grid_size/3), n_bandwidths)

  results_dict = {'grid_size': grid_size,
                  'cov': cov_name,
                  'betas':
                    {float(beta_): {
                     'sigma_sq_infty_closed_form': None,
                      'bandwidths': {float(bandwidth_): {'mse': None, 'coverage': None} for bandwidth_ in bandwidths},
                    } for beta_ in betas}
                  }

  # Compute mse for each covariance parameter beta and bandwidth
  for beta in betas:
    print('beta: {}'.format(beta))
    if cov_name == 'exponential':
      cov, root_cov = get_exponential_gaussian_covariance(beta1=beta, beta2=beta, grid_size=grid_size)
    # ToDo: shouldn't have to pass any covariance if identity
    elif cov_name == 'identity':
      cov, root_cov = np.eye(grid_size), np.eye(grid_size)
    sigma_sq_infty_closed_form = cov[0, :].sum()
    results_dict['betas'][beta]['sigma_sq_infty_closed_form'] = float(sigma_sq_infty_closed_form)
    for b in bandwidths:
      print('bandwidth: {}'.format(b))
      kernel = lambda k: block(k, b)
      mse, coverage = get_var_estimate_mse(kernel=kernel, sigma_sq_infty=sigma_sq_infty_closed_form,
                                           root_cov=root_cov, pairwise_distances=pairwise_distances, n_rep=n_rep,
                                           pct_cores=pct_cores)
      results_dict['betas'][beta]['bandwidths'][b]['mse'] = float(mse)
      results_dict['betas'][beta]['bandwidths'][b]['coverage'] = float(coverage)

  # Save results
  prefix = os.path.join(this_dir, 'variance_estimates')
  info = 'cov_name={}-size={}'.format(cov_name, grid_size)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  filename = '{}/{}_{}.yml'.format(prefix, info, suffix)
  with open(filename, 'w') as outfile:
    yaml.dump(results_dict, outfile)

  return


def var_sigma_infty_from_exp_kernel(beta1=0.1, beta2=0.1, grid_size=100):
  cov, _ = get_exponential_gaussian_covariance(beta1=beta1, beta2=beta2, grid_size=grid_size)
  sigma_sq_infty = cov[0, :].sum()
  squared_residuals_sum = 2*(cov[0, :]**2).sum()
  print('sigma sq infty: {} sq resid sum: {}'.format(sigma_sq_infty, squared_residuals_sum))
  return


def backup_sampling_dbn_rep(seed, time_horizon, n_cutoff, kernel, beta1, beta2, N, c1, c2):

  np.random.seed(seed)
  pairwise_distances = get_pairwise_distances(grid_size)
  spatial_kernel_weights = construct_kernel_matrix_from_distances(kernel, pairwise_distances)
  temporal_kernel_weights = np.array([np.array([kernel(np.abs(t1 - t2)) for t1 in range(time_horizon)])
                                      for t2 in range(time_horizon)])
  _, root_cov = get_exponential_gaussian_covariance(grid_size=grid_size, beta1=beta1, beta2=beta2)
  identity_root_cov = np.eye(grid_size)
  spatiotemporal_kernel_weights = get_spatiotemporal_kernel(spatial_kernel_weights, temporal_kernel_weights, grid_size,
                                                            N)

  y = np.zeros(0)
  X = np.zeros((0, 2))

  # Generate data
  x = generate_gaussian(identity_root_cov)
  for t in range(time_horizon):
    x_cutoff = np.sort(x)[n_cutoff]
    x_indicator = (x > x_cutoff)
    errors = generate_gaussian(root_cov)
    x_new = c1 * x + c2 * x_indicator + errors

    # Append to dataset
    y = np.hstack((y, x_new))
    X_t = np.column_stack((x, x_indicator))
    X = np.vstack((X, X_t))

    x = x_new

  # 0-step q-function
  Xprime_X = np.dot(X.T, X)
  Xprime_X_inv = np.linalg.inv(Xprime_X)
  Xy = np.dot(X.T, y)
  beta0_hat = np.dot(Xprime_X_inv, Xy)

  # Backup
  X0 = X[:-1, :]
  X1 = X[1:, :]
  V_hat = np.dot(X1, beta0_hat)
  q1 = X1[:, 0] + V_hat

  # 1-step q-function
  Xq = np.dot(X0.T, q1)
  beta1_hat = np.dot(Xprime_X_inv, Xq)

  # Estimate covariance and construct CI
  X_times_q = np.multiply(X0, q1[:, np.newaxis])
  X_times_q = X_times_q - X_times_q.mean(axis=0)
  inner_cov_hat = estimate_spatiotemporal_matrix_var(X_times_q, spatiotemporal_kernel_weights)
  cov_hat = grid_size * np.dot(Xprime_X_inv, np.dot(inner_cov_hat, Xprime_X_inv))
  beta1_1_hat = beta1_hat[1]
  beta1_1_var_hat = cov_hat[1, 1]
  ci_upper = beta1_1_hat + 1.96 * np.sqrt(beta1_1_var_hat)
  ci_lower = beta1_1_hat - 1.96 * np.sqrt(beta1_1_var_hat)
  return {'ci_lower': ci_lower, 'ci_upper': ci_upper, 'Xq': Xq, 'beta1_hat': beta1_hat, 'Xprime_X': Xprime_X}


def backup_sampling_dbn(grid_size, bandwidth, kernel_name='bartlett', beta1=1, beta2=1, n_rep=100, pct_treat=0.1,
                        time_horizon=10):
  """
  Calculate sampling dbn of 1-step Q-function, where

  X ~ N(0, Cov)
  Y_i = c1*X_i + c2*1[ X_i > X_{(1-pct_treat)*L}
  """
  c1, c2 = 0.5, 1.

  # Construct kernel weight matrix
  if kernel_name == 'bartlett':
    kernel = partial(bartlett, bandwidth=bandwidth)
  elif kernel_name == 'delta':
    def kernel(x): return 0
  elif kernel_name == 'block':
    kernel = partial(block, bandwidth=bandwidth)

  n_cutoff = int((1 - pct_treat) * grid_size)
  beta1_hat_dbn = np.zeros((0, 2))
  Xprime_X_lst = np.zeros((n_rep, 2, 2))
  Xq_lst = np.zeros((n_rep, 2))
  ci_lst = np.zeros((n_rep, 2))
  N = time_horizon * grid_size

  # Distribute
  backup_sampling_dbn_partial = partial(backup_sampling_dbn_rep, time_horizon=time_horizon, n_cutoff=n_cutoff,
                                        kernel=kernel, beta1=beta1, beta2=beta2, N=N, c1=c2, c2=c2)
  if n_rep == 1:
    results = [backup_sampling_dbn_partial(0)]
  else:
    pool = mp.Pool(processes=4)
    results = pool.map(backup_sampling_dbn_partial, range(n_rep))

  for n, res in enumerate(results):
    ci_lower = res['ci_lower']
    ci_upper = res['ci_upper']
    Xq = res['Xq']
    beta1_hat = res['beta1_hat']
    Xprime_X = res['Xprime_X']
    beta1_hat_dbn = np.vstack((beta1_hat_dbn, beta1_hat))
    Xprime_X_lst[n, :] = Xprime_X / grid_size
    Xq_lst[n, :] = Xq
    ci_lst[n, :] = [ci_lower, ci_upper]

  Xprime_X_true = np.mean(Xprime_X_lst, axis=0) * grid_size
  Xprime_X_inv_true = np.linalg.inv(Xprime_X_true)
  Xq_true = np.mean(Xq_lst, axis=0)
  beta_true = np.dot(Xprime_X_inv_true, Xq_true)
  beta1_true = beta_true[1]
  contains_truth = np.multiply(beta1_true < ci_lst[:, 1], beta1_true > ci_lst[:, 0])
  coverage = np.mean(contains_truth)

  # c2_population_var_hat = np.var(c_dbn[:, 1])
  return beta1_hat_dbn, Xprime_X_lst, coverage


def simple_action_sampling_dbn(grid_size, bandwidth, kernel_name='bartlett', beta1=1, beta2=1, n_rep=100, pct_treat=0.1,
                               time_horizon=10, heteroskedastic=False):
  """
  Treat pct_treat largest observations.

  X ~ N(0, Cov)
  Y_i = c1*X_i + c2*1[ X_i > X_{(1-pct_treat)*L}

  Calculate sampling dbn of [c1_hat, c2_hat].
  """
  c1, c2 = 0.5, 1.

  # Construct kernel weight matrix
  if kernel_name == 'bartlett':
    kernel = partial(bartlett, bandwidth=bandwidth)
  elif kernel_name == 'delta':
    def kernel(x): return 0
  elif kernel_name == 'block':
    kernel = partial(block, bandwidth=bandwidth)

  pairwise_distances = get_pairwise_distances(grid_size)
  spatial_kernel_weights = construct_kernel_matrix_from_distances(kernel, pairwise_distances)
  temporal_kernel_weights = np.array([np.array([kernel(np.abs(t1 - t2)) for t1 in range(time_horizon)])
                                      for t2 in range(time_horizon)])

  n_cutoff = int((1-pct_treat)*grid_size)
  _, root_cov = get_exponential_gaussian_covariance(grid_size=grid_size, beta1=beta1, beta2=beta2,
                                                    heteroskedastic=heteroskedastic)
  identity_root_cov = np.eye(grid_size)
  c_dbn = np.zeros((0, 2))
  Xprime_X_lst = np.zeros((n_rep, 2, 2))
  coverage = 0.
  chat_var_lst = []
  N = time_horizon * grid_size
  spatiotemporal_kernel_weights = get_spatiotemporal_kernel(spatial_kernel_weights, temporal_kernel_weights, grid_size,
                                                            N)

  # For assessing alpha-mixing coef of residuals at l, lprime, conditional on actions
  e_l_lst = np.zeros(n_rep)
  e_lprime_lst = np.zeros(n_rep)
  x_indicator_lst = np.zeros(0)
  X_lst = []
  y_lst = []

  for n in range(n_rep):
    y = np.zeros(0)
    X = np.zeros((0, 2))

    # Generate data
    x = generate_gaussian(identity_root_cov)
    for t in range(time_horizon):
      x_cutoff = np.sort(x)[n_cutoff]
      x_indicator = (x > x_cutoff)
      errors = generate_gaussian(root_cov)
      x_new = c1*x + c2*x_indicator + errors

      # Append to dataset
      y = np.hstack((y, x_new))
      x_indicator_lst = np.hstack((x_indicator_lst, x_indicator))
      X_t = np.column_stack((x, x_indicator))
      X = np.vstack((X, X_t))

      x = x_new

    X_lst.append(X)
    y_lst.append(y)

    # Regression
    Xprime_X = np.dot(X.T, X)
    Xprime_X_inv = np.linalg.inv(Xprime_X)
    Xy = np.dot(X.T, y)
    c_hat = np.dot(Xprime_X_inv, Xy)

    c_dbn = np.vstack((c_dbn, c_hat))
    Xprime_X_lst[n, :] = Xprime_X / grid_size

    # Estimate covariance and construct CI
    X_times_y = np.multiply(X, y[:, np.newaxis])
    X_times_y = X_times_y - X_times_y.mean(axis=0)
    inner_cov_hat = estimate_spatiotemporal_matrix_var(X_times_y, spatiotemporal_kernel_weights)
    cov_hat = grid_size*np.dot(Xprime_X_inv, np.dot(inner_cov_hat, Xprime_X_inv))
    c2_hat = c_hat[1]
    c2_var_hat = cov_hat[1, 1]
    ci_upper = c2_hat + 1.96*np.sqrt(c2_var_hat)
    ci_lower = c2_hat - 1.96*np.sqrt(c2_var_hat)
    contains_truth = (ci_lower < c2 < ci_upper)
    coverage += contains_truth / n_rep
    chat_var_lst.append(c2_var_hat)

  # Residual joint distribution (for assessing alpha-mixing)
  c_true = np.array([c1, c2])
  yHat_lst = [np.dot(X_, c_true) for X_ in X_lst]
  for rep in range(n_rep):
    # Condition on x_indicator_l_t, x_indicator_lprime_tp1
    x_indicator_rep = X_lst[rep][:, 1]
    x_indicator_t = x_indicator_rep[(1 * grid_size):(2 * grid_size)]
    x_indicator_tp1 = x_indicator_rep[(2 * grid_size):(3 * grid_size)]
    if x_indicator_t[0] == 0 and x_indicator_tp1[-1] == 0:
      yHat_t = yHat_lst[rep][(1*grid_size):(2*grid_size)]
      y_t = y_lst[rep][(1*grid_size):(2*grid_size)]
      yHat_tp1 = yHat_lst[rep][(2 * grid_size):(3 * grid_size)]
      y_tp1 = y_lst[rep][(2 * grid_size):(3 * grid_size)]

      residual_t_l = yHat_t[0] - y_t[0]
      residual_t_lprime = yHat_tp1[-1] - y_tp1[-1]

      e_l_lst[rep] = residual_t_l
      e_lprime_lst[rep] = residual_t_lprime

  # c2_population_var_hat = np.var(c_dbn[:, 1])
  pvals = normaltest(c_dbn, axis=0)[1]
  return c_dbn, Xprime_X_lst, coverage, pvals, e_l_lst, e_lprime_lst


def regress_on_summary_statistic():
  pct_treat = 0.1
  z_list = []
  n_rep = 100
  for grid_size in [100, 1000, 10000]:
    y_cutoffs = np.zeros(n_rep)
    z_list_grid_size = np.zeros(n_rep)
    n_cutoff = int((1 - pct_treat) * grid_size)
    _, root_cov = get_exponential_gaussian_covariance(beta1=1, beta2=1, grid_size=grid_size)
    for n in range(n_rep):
      y = generate_gaussian(root_cov)
      y_cutoff = np.sort(y)[n_cutoff]
      y_cutoffs[n] = y_cutoff
      z = np.multiply(y, y > y_cutoff)
      z_list_grid_size[n] = np.mean(z)
    plt.scatter(y_cutoffs, z_list_grid_size, label=str(grid_size))
  plt.legend()
  plt.show()


def alpha_cdf(x, y, u):
  cdf_x = np.mean(x < u)
  cdf_y = np.mean(y < u)
  cdf_joint = np.mean(np.multiply(x < u, y < u))
  cdf_ind = cdf_x * cdf_y
  alpha = np.abs(cdf_ind - cdf_joint)
  return alpha


def test_alpha_mixing():
  kernel_name = 'block'
  beta = 1.0
  heteroskedastic = True
  bandwidth = 3
  u = 0.1
  bootstrap_reps = 100
  for grid_size in [25, 64, 100, 225, 400]:
    _, _, coverage, pvals, e_l_lst, e_lprime_lst = \
      simple_action_sampling_dbn(grid_size=grid_size, bandwidth=bandwidth,
                                 kernel_name=kernel_name,
                                 beta1=beta, beta2=beta, n_rep=1000,
                                 pct_treat=0.1, heteroskedastic=heteroskedastic)

    # Get CDFs to assess alpha-mixing
    alpha = alpha_cdf(e_l_lst, e_lprime_lst, u)

    # Bootstrap estimate of standard error
    sample_size = len(e_l_lst)
    alpha_boot_lst = []
    for b in range(bootstrap_reps):
      boot_ixs = np.random.choice(sample_size, size=sample_size, replace=True)
      e_l_boot = e_l_lst[boot_ixs]
      e_lprime_boot = e_lprime_lst[boot_ixs]
      alpha_boot = alpha_cdf(e_l_boot, e_lprime_boot, u)
      alpha_boot_lst.append(alpha_boot)

    se = np.std(alpha_boot_lst)
    # _, _, coverage = backup_sampling_dbn(grid_size, bandwidth, kernel_name='bartlett', beta1=1, beta2=1, n_rep=100,
    #                                      pct_treat=0.1, time_horizon=10)
    print('bandwidth: {} coverage: {} pvals: {} alpha: {} se: {}'.format(bandwidth, coverage, pvals, alpha, se))


if __name__ == "__main__":
  # ToDo: write big matrices to file so that they don't have to be re-computed (and that they can work with
  # ToDo: multiprocessing?

  parser = argparse.ArgumentParser()
  parser.add_argument('--n_rep', type=int)
  parser.add_argument('--grid_size', type=int)
  args = parser.parse_args()

  kernel_name = 'block'
  beta = 1.0
  heteroskedastic = True
  grid_size = args.grid_size
  bandwidths = np.linspace(5, 30, 10)
  for bandwidth in bandwidths:
    beta1_hat_dbn, Xprime_X_lst, coverage = \
      backup_sampling_dbn(grid_size, bandwidth, kernel_name='bartlett', beta1=1, beta2=1, n_rep=args.n_rep, pct_treat=0.1,
                          time_horizon=5)

    print('bandwidth: {} coverage: {}'.format(bandwidth, coverage))
