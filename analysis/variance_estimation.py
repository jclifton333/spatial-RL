import numpy as np
import scipy.linalg as la
import pdb
import multiprocessing as mp
from functools import partial
import yaml
import datetime

import os
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_pairwise_distances(grid_size):
  length = np.floor(np.sqrt(grid_size)).astype(int)
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


def get_exponential_gaussian_covariance(beta1=1, beta2=2, grid_size=100):
  """
  Using gen model from "Optimal block size for variance estimation by a spatial block bootstrap method".

  :param grid_size: Will sample observations on floor(sqrt(grid_size)) x floor(sqrt(grid_size)) lattice
  """
  # Get covariance matrix
  length = np.floor(np.sqrt(grid_size)).astype(int)
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

  # Products of centered residuals
  y_centered = y - np.mean(y)
  cross_raw = np.outer(y_centered, y_centered)

  cross_kernel = np.multiply(cross_raw, kernel_weights)
  sigma_sq_hat = cross_kernel.sum() / grid_size

  return y, sigma_sq_hat


def get_var_estimate_mse(kernel, root_cov, pairwise_distances, n_rep=10000, pct_cores=0.25):
  var_estimates = np.zeros(0)
  grid_size = root_cov.shape[0]
  observations = np.zeros((0, grid_size))

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

  # autocovs = get_autocov_sq_sequence(observations, pairwise_distances)
  sigma_sq_infty, sq_residual_cov = get_sigma_sq_infty(observations)
  mse_kernel = np.mean((var_estimates - sigma_sq_infty)**2)
  print('sq residual cov: {}'.format(sq_residual_cov))
  print('mse: {}'.format(mse_kernel))
  return mse_kernel


def bartlett(x, bandwidth):
  x_abs = np.abs(x) / bandwidth
  k = (1 - x_abs) * (x_abs <= 1)
  return k


def constant(x):
  return 1


def var_estimates(n_bandwidths=10, betas=(0.1,), grid_size=100, n_rep=1000, pct_cores=0.25):
  """
  :param n_bandwidths: Bandwidths will be n_bandwidths values log-spaced between 1 and max(pairwise_distances).
  """

  pairwise_distances = get_pairwise_distances(grid_size)
  max_dist = np.max(pairwise_distances)
  bandwidths = np.logspace(np.log10(1.), np.log10(max_dist), n_bandwidths)

  results_dict = {'grid_size': grid_size,
                  'betas':
                    {float(beta_): {
                     'sigma_sq_infty_closed_form': None,
                      'bandwidth_mses': {float(bandwidth_): None for bandwidth_ in bandwidths}
                    } for beta_ in betas}
                  }

  # Compute mse for each covariance parameter beta and bandwidth
  for beta in betas:
    print('beta: {}'.format(beta))
    cov, root_cov = get_exponential_gaussian_covariance(beta1=beta, beta2=beta, grid_size=grid_size)
    sigma_sq_infty_closed_form = cov[0, :].sum()
    results_dict['betas'][beta]['sigma_sq_infty_closed_form'] = float(sigma_sq_infty_closed_form)
    for b in bandwidths:
      print('bandwidth: '.format(b))
      kernel = lambda k: bartlett(k, b)
      mse = get_var_estimate_mse(kernel=kernel, root_cov=root_cov, pairwise_distances=pairwise_distances, n_rep=n_rep,
                                 pct_cores=pct_cores)
      results_dict['betas'][beta]['bandwidth_mses'][b] = float(mse)

  # Save results
  prefix = os.path.join(this_dir, 'variance_estimates')
  info = 'size={}'.format(grid_size)
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


if __name__ == "__main__":
  var_estimates(grid_size=100, n_rep=10000)
  var_estimates(grid_size=1600, n_rep=10000)
  var_estimates(grid_size=10000, n_rep=10000)

