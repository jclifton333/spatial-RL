import numpy as np
import scipy.linalg as la
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
def get_exponential_gaussian_covariance_helper(beta1=1, beta2=2, grid_size=100):
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
          cov[index_1, index_2] = np.exp(-weighted_distance)

  return cov


def get_exponential_gaussian_covariance(beta1=1, beta2=1, grid_size=100):
  cov = get_exponential_gaussian_covariance_helper(beta1=beta1, beta2=beta2, grid_size=grid_size)
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


def estimate_matrix_var(Y, kernel_weights):
  # Estimate covariance from a list of vectors Y
  Y_bar = np.mean(Y, axis=0)
  Y_centered = Y - Y_bar
  L, p = Y.shape

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


def get_var_estimate_mse(kernel, sigma_sq_infty, root_cov, pairwise_distances, n_rep=10000, pct_cores=0.25):
  var_estimates = np.zeros(0)
  grid_size = root_cov.shape[0]
  observations = np.zeros((0, grid_size))
  confidence_intervals = np.zeros((0, 2))

  # Construct kernel weight matrix
  kernel_weights = np.zeros((grid_size, grid_size))
  for i in range(grid_size):
    for j in range(grid_size):
      kernel_weights[i, j] = kernel(pairwise_distances[i, j])

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
      kernel = lambda k: bartlett(k, b)
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


def simple_action_sampling_dbn(grid_size, kernel_weights, beta1=1, beta2=1, n_rep=100, pct_treat=0.1):
  """
  Treat pct_treat largest observations.

  X ~ N(0, Cov)
  Y_i = c1*X_i + c2*1[ X_i > X_{(1-pct_treat)*L}

  Calculate sampling dbn of [c1_hat, c2_hat].
  """
  c1, c2 = 1., 1.

  n_cutoff = int((1-pct_treat)*grid_size)
  _, root_cov = get_exponential_gaussian_covariance(grid_size=grid_size, beta1=beta1, beta2=beta2)
  c_dbn = np.zeros((0, 2))
  Xprime_X_lst = np.zeros((n_rep, 2, 2))
  coverage = 0.

  for n in range(n_rep):
    # Generate data
    x = generate_gaussian(root_cov)
    x_cutoff = np.sort(x)[n_cutoff]
    x_indicator = (x > x_cutoff)
    y = c1*x + c2*x_indicator + np.random.normal(scale=0.5, size=grid_size)

    # Regression
    X = np.column_stack((x, x_indicator))
    Xprime_X = np.dot(X.T, X)
    Xprime_X_inv = np.linalg.inv(Xprime_X)
    Xy = np.dot(X.T, y)
    c_hat = np.dot(Xprime_X_inv, Xy)

    c_dbn = np.vstack((c_dbn, c_hat))
    Xprime_X_lst[n, :] = Xprime_X / grid_size

    # Estimate covariance and construct CI
    inner_cov_hat = estimate_matrix_var(Xy, kernel_weights)
    cov_hat = np.dot(Xprime_X_inv, np.dot(inner_cov_hat, Xprime_X_inv))
    c2_hat = c_hat[1]
    c2_var_hat = cov_hat[1, 1]
    ci_upper = c2_hat + 1.96*c2_var_hat/np.sqrt(grid_size)
    ci_lower = c2_hat - 1.96*c2_var_hat/np.sqrt(grid_size)
    contains_truth = (ci_lower < c2 < ci_upper)
    coverage += contains_truth / n_rep

  return c_dbn, Xprime_X_lst, coverage


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


if __name__ == "__main__":
  # var_estimates(cov_name='exponential', grid_size=100, n_rep=100)
  var_estimates(cov_name='exponential', grid_size=900, n_rep=5000)
  var_estimates(cov_name='exponential', grid_size=1600, n_rep=5000)
  var_estimates(cov_name='exponential', grid_size=6400, n_rep=5000)
  # get_exponential_gaussian_covariance(beta1=1, beta2=2, grid_size=6400)
  # get_pairwise_distances(6400)
  # grid_size = 900
  # beta = 0.1
  # c_dbn, Xprime_Xs = simple_action_sampling_dbn(grid_size, beta1=beta, beta2=beta, n_rep=100, pct_treat=0.1)



