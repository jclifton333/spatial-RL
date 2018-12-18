"""
Spread of disease in continuous space (pdf pg 13 of white nose paper);
locations are points in [0,1]^2.

This is still just the gravity model, though.
"""
import numpy as np
import src.environments.SpatialDisease import SpatialDisease


class Continuous(SpatialDisease):
  COVARIANCE_KERNEL_PARAMETERS = np.ones(4)

  def __init__(self, L):
    adjacency_matrix = np.ones((L, L))  # Fully connected

    # Generate locations and pairwise distances
    self.location_coordinates = np.random.random(size=(L, L))
    self.distance_matrix = np.array([
      np.array([
        np.linalg.norm(x_l - x_lprime) for x_l in self.location_coordinates
      ])
    for x_lprime in self.location_coordinates])
    self.distance_matrix /= np.std(self.distance_matrix)

    # Generate static covariates
    # From paper: For each location l, we generate four static covariates by using a mean 0 Gaussian process
    # with a multivariate separable isotropic covariance matrix that is exponential space and
    # autoregressive across the four covariates at each location.
    covariance_matrices = np.array([np.array([
      self.covariate_covariance(l, lprime) for l in range(L)
    ]) for lprime in range(L)])

    # ToDo: Make sure this is what is meant by autoregressive...
    s_1 = np.random.multivariate_normal(np.zeros(L), covariance_matrices[:, :, 0])
    s_2 = np.random.multivariate_normal(s_1, covariance_matrices[:, :, 1])
    s_3 = np.random.multivariate_normal(s_2, covariance_matrices[:, :, 2])
    s_4 = np.random.multivariate_normal(s_3, covariance_matrices[:, :, 3])
    z = s_1 - np.min(s_1)
    y = np.random.binomial(1, 0.01, L)
    self.current_state = np.column_stack((s_1, s_2, s_3, s_4, z, y))



    SpatialDisease.__init__(self, adjacency_matrix)

  def covariate_covariance(self, l, lprime):
    covs_for_each_dimension = []
    x_l, x_lprime = self.location_coordinates[l, :], self.location_coordinates[lprime, :]
    squared_dist = np.dot(x_l - x_lprime, x_l - x_lprime)
    for parameter in Continuous.COVARIANCE_KERNEL_PARAMETERS:
      covs_for_each_dimension.append(
        np.exp(-parameter * squared_dist / 2)
      )
    return np.array(covs_for_each_dimension)

  def generate_raw_covariates(self):

  def next_state(self):
    pass
