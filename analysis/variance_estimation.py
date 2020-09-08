import numpy as np
import scipy.linalg as la
import pdb


def get_exponential_gaussian_covariance(beta1=1, beta2=2, grid_size=100):
  """
  Using gen model from "Optimal block size for variance estimation by a spatial block bootstrap method".

  :param grid_size: Will sample observations on floor(sqrt(grid_size)) x floor(sqrt(grid_size)) lattice
  """
  # Get pairwise distances and covariance matrix
  length = np.floor(np.sqrt(grid_size)).astype(int)
  true_grid_size = length**2
  pairwise_distances = np.zeros((true_grid_size, true_grid_size))
  cov = np.zeros((true_grid_size, true_grid_size))
  for i in range(length):
    for j in range(length):
      for iprime in range(length):
        for jprime in range(length):
          index_1 = length*i + j
          index_2 = length*iprime + jprime
          distance = np.abs(i - iprime) + np.abs(j - jprime)
          weighted_distance = beta1*np.abs(i - iprime) + beta2*np.abs(j - jprime)
          pairwise_distances[index_1, index_2] = distance
          cov[index_1, index_2] = np.exp(-weighted_distance)

  root_cov = la.sqrtm(cov)
  return root_cov, pairwise_distances


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
  max_autocov_sq_sequence = np.zeros(max_dist)

  # Need mean squared residuals for computing covariances between squared residuals
  sample_means = observations.mean(axis=1)
  squared_residuals = (observations - sample_means)**2
  mean_squared_residuals = squared_residuals.mean(axis=0)
  centered_squared_residuals = squared_residuals - mean_squared_residuals

  # Loop through distances and get the max covariance between location 0 and locations at that distance
  res_sq_0 = centered_squared_residuals[:, 0]
  for i in range(length):
    for j in range(length):
      d = i + j - 1
      ix = length*i + j
      res_sq_ix = centered_squared_residuals[:, ix]
      autocov = np.abs(np.dot(res_sq_0, res_sq_ix) / n_rep)
      if autocov > max_autocov_sq_sequence[d]:
        max_autocov_sq_sequence[d] = autocov

  return max_autocov_sq_sequence


if __name__ == "__main__":
  n_rep = 100
  grid_size = 100
  naive_var_estimates = np.zeros(0)
  observations = np.zeros((0, grid_size))
  beta1 = 10
  beta2 = 10
  root_cov, pairwise_distances = get_exponential_gaussian_covariance(beta1=beta1, beta2=beta2, grid_size=grid_size)

  for _ in range(n_rep):
    y = generate_gaussian(root_cov)
    sigma_sq_hat = np.var(y)
    naive_var_estimates = np.hstack((naive_var_estimates, sigma_sq_hat))
    observations = np.vstack((observations, y))

  autocovs = get_autocov_sq_sequence(observations, pairwise_distances)












