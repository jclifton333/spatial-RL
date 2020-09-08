import numpy as np
import scipy.linalg as la


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
          distance = beta1*np.abs(i - iprime) + beta2*np.abs(j - jprime)
          pairwise_distances[index_1, index_2] = distance
          cov[index_1, index_2] = np.exp(-distance)

  root_cov = la.sqrtm(cov)
  return root_cov, pairwise_distances


def generate_gaussian(root_cov):
  n = root_cov.shape[0]
  y_raw = np.random.normal(size=n)
  y = np.dot(root_cov, y_raw)
  return y


if __name__ == "__main__":
  n_rep = 100
  grid_size = 100
  naive_var_estimates = np.zeros(0)
  observations = np.zeros((0, grid_size))
  root_cov, _ = get_exponential_gaussian_covariance(grid_size)

  for _ in range(n_rep):
    y = generate_gaussian(root_cov)
    sigma_sq_hat = np.var(y)
    naive_var_estimates = np.hstack((naive_var_estimates, sigma_sq_hat))
    observations = np.vstack((observations, y))

  pop_var = observations.var(axis=0)













