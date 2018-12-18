import numpy as np
from .ebola_infection_probs import expit2


def transmission_prob(a, x, l, l_prime, eta, L, **kwargs):
  """
  Equation 2 of white nose paper. eta in our notation =
  [theta_0, theta_1 (vector), theta_2 (vector), theta_3, theta_4, theta_5, theta_6]

  :param a:
  :param x:
  :param l:
  :param l_prime:
  :param eta:
  :param L:
  :param kwargs:
  :return:
  """
  distance_matrix, z = kwargs['distance_matrix'], kwargs['z']
  theta_0 = eta[0]
  theta_1 = eta[1:5]
  theta_2 = eta[5:9]
  theta_3 = eta[9]
  theta_4 = eta[10]
  theta_5 = eta[11]
  theta_6 = eta[12]

  d_l_lprime = distance_matrix[l, l_prime]
  x_l = x[l, :]
  x_lprime = x[l_prime, :]
  logit = theta_0 + np.dot(theta_1, x_l) + np.dot(theta_2, x_lprime) - theta_3*a[l] - theta_4*a[l_prime] - \
    theta_5 * d_l_lprime / np.power(z[l] * z[l_prime], theta_6)
  transmission_prob_ = expit2(logit)
  return transmission_prob_


