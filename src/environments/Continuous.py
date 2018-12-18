"""
Spread of disease in continuous space (pdf pg 13 of white nose paper);
locations are points in [0,1]^2.

This is still just the gravity model, though.
"""
import numpy as np
from scipy.special import expit
import src.environments.Gravity import Gravity
import src.environments.gravity_infection_probs as infection_probs

# ToDo: Make Gravity superclass from which Continuous, Ebola inherit!
class Continuous(SpatialDisease):
  # ToDo: These are placeholders!
  COVARIANCE_KERNEL_PARAMETERS = np.ones(4)
  THETA_0 = 0.0
  THETA_1 = np.ones(4)
  THETA_2 = np.ones(4)
  THETA_3 = THETA_4 = THETA_5 = THETA_6 = 1.0

  def __init__(self, L):
    adjacency_matrix = np.ones((L, L))  # Fully connected

    # Generate locations and pairwise distances
    self.location_coordinates = np.random.random(size=(L, L))
    distance_matrix = np.array([
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

    # Compute transmission_probs

    # ToDo: Make sure this is what is meant by autoregressive...
    s_1 = np.random.multivariate_normal(np.zeros(L), covariance_matrices[:, :, 0])
    s_2 = np.random.multivariate_normal(s_1, covariance_matrices[:, :, 1])
    s_3 = np.random.multivariate_normal(s_2, covariance_matrices[:, :, 2])
    s_4 = np.random.multivariate_normal(s_3, covariance_matrices[:, :, 3])
    self.z = s_1 - np.min(s_1)
    product_matrix = np.outer(z, z)
    covariate_matrix = np.column_stack((s_1, s_2, s_3, s_4))
    self.set_transmission_probs()
    self.current_infected = np.random.binomial(1, 0.01, L)
    self.current_state = np.column_stack((s_1, s_2, s_3, s_4, z, self.current_infected))
    Gravity.__init__(self, distance_matrix, product_matrix, adjacency_matrix, covariate_matrix)

  def set_transmission_probs(self):
    self.transmission_probs = np.zeros((self.L, self.L, 2, 2))
    for l in range(self.L):
      x_l = self.x[l, :]
      for lprime in range(self.L):
        x_lprime = self.x[lprime, :]
        d_l_lprime = self.distance_matrix[l, lprime]
        z_l_lprime = self.product_matrix[l, lprime]
        baseline_logit = self.THETA_0 + np.dot(self.THETA_1, x_l) + np.dot(self.THETA_2, x_lprime) - \
          self.THETA_5 * d_l_lprime / np.power(z_l_lprime, self.THETA_6)
        self.transmission_probs[l, lprime, 0, 0] = expit(baseline_logit)
        self.transmission_probs[l, lprime, 1, 0] = expit(baseline_logit - Continuous.THETA_3)
        self.transmission_probs[l, lprime, 0, 1] = expit(baseline_logit - Continuous.THETA_4)
        self.transmission_probs[l, lprime, 1, 1] = expit(baseline_logit - Continuous.THETA_3 - Continuous.THETA_4)

  def covariate_covariance(self, l, lprime):
    covs_for_each_dimension = []
    x_l, x_lprime = self.location_coordinates[l, :], self.location_coordinates[lprime, :]
    squared_dist = np.dot(x_l - x_lprime, x_l - x_lprime)
    for parameter in Continuous.COVARIANCE_KERNEL_PARAMETERS:
      covs_for_each_dimension.append(
        np.exp(-parameter * squared_dist / 2)
      )
    return np.array(covs_for_each_dimension)

  def reset(self):
    super(Continuous, self).reset()

  def transmission_prob(self, a, l, lprime, eta):
    if self.current_infected[lprime]:
      if eta is None:
        transmission_prob = self.transmission_probs[l, lprime, a[l], a[lprime]]
      else:
        transmission_prob = infection_probs.continuous_transmission_probs(l, lprime, a, eta, self.distance_matrix,
                                                                          self.z, self.x)
      return transmission_prob
    else:
      return 0.0

  def infection_prob_at_location(self, a, l, eta):
    if self.current_infected[l]:
      return 1.0
    else:
      not_transmitted_prob = np.product(
        [1 - self.transmission_prob(a, l, l_prime, eta) for l_prime in self.adjacency_list[l]])
      inf_prob = 1 - not_transmitted_prob
      return inf_prob

  def next_state(self):
    pass
