
"""
Spread of disease in continuous space (pdf pg 13 of white nose paper);
locations are points in [0,1]^2.

This is still just the gravity model, though.
"""
import numpy as np
from scipy.special import expit
import src.environments.SpatialDisease import SpatialDisease
import src.environments.gravity_infection_probs as infection_probs


class Gravity(SpatialDisease):
  """
  Class for gravity disease models.  Ebola and the generic Continuous class will inherit from this.

  Transmission probabilities in gravity model are of the form

  logit(p_l_lprime) = \theta_0 + <\theta_1 | x_l> + <\theta_2 | x_lprime> - \theta_3*a_l - \theta_4*a_l
    - \theta_5 * d_l_lprime / (product_l_lprime)**\theta_6

  where (x_l, x_lprime) are optional covariates (Ebola doesn't have any).
  """
  def __init__(self, distance_matrix, product_matrix, covariate_matrix):
    self.distance_matrix = distance_matrix
    self.product_matrix = product_matrix
    self.covariate_matrix = covariate_matrix
    self.L = distance_matrix.shape[0]

  def transmission_prob(self, a, l, lprime, eta):
    if self.current_infected[lprime]:
      if eta is None:
        transmission_prob = self.transmission_probs[l, lprime, int(a[l]), int(a[lprime])]
      else:
        transmission_prob = infection_probs.transmission_prob(a, l, lprime, eta, self.covariate_matrix,
                                                              self.distance_matrix, self.product_matrix)
    else:
      transmission_prob = 0.0
    return transmission_prob

  def infection_prob_at_location(self, a, l, eta):
    if self.current_infected[l]:
      return 1.0
    else:
      not_transmitted_prob = np.product([1 - self.transmission_prob(a, l, lprime, eta) for lprime in
                                         self.adjacency_list[l]])
      inf_prob = 1 - not_transmitted_prob
      return inf_prob

  def next_infected_probabilities(self, a, eta=None):
    return np.array([
      self.infection_prob_at_location(a, l, eta) for l in range(self.L)
    ])

  def feature_function(self, raw_data_block):
    pass

  def feature_function_at_action(self, old_data_block, old_action, action):
    pass

  def feature_function_at_location(self, l, raw_data_block):
    pass

  def update_obs_history(self, a):
    super(Gravity, self).update_obs_history(a)
    raw_data_block = np.column_stack((self.covariate_matrix, a, self.Y[-2, :]))
    data_block = self.feature_function(raw_data_block)
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)

  def next_state(self):
    super(Gravity, self).next_state()

  def next_infections(self, a, eta=None):
    next_infected_probabilities = self.next_infected_probabilities(a, eta=eta)
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.current_infected = next_infections

  def data_block_at_action(self, data_block_ix, action, raw=False):
    super(Gravity, self).data_block_at_action(data_block_ix, action)
    if raw:
      new_data_block = copy.copy(self.X_raw[data_block_ix])
      new_data_block[:, 1] = action
    else:
      new_data_block = self.feature_function_at_action(self.X[data_block_ix], self.A[data_block_ix, :], action)
    return new_data_block

  def mb_covariance(self, mb_params):
    """
    Covariance of gravity model MLE.

    :param mb_params: MLE of \eta.
    :return:
    """



