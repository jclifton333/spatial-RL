import numpy as np
from scipy.special import expit
from numba import njit


@njit
def expit2(x):
  """
  To use with njit.

  :param x:
  :return:
  """
  exp_ = np.exp(-x)
  return 1.0 - 1.0 / (1 + exp_)


def ebola_infection_probs(a, y, eta, L, adjacency_lists, **kwargs):
  distance_matrix, susceptibility = kwargs['distance_matrix'], kwargs['susceptibility']
  return np.array([infection_prob_at_location(a, l, eta, y, adjacency_lists, distance_matrix,
                                              susceptibility) for l in range(L)])


def transmission_prob(a, l, l_prime, eta, distance_matrix, susceptibility):
  d_l_lprime = distance_matrix[l, l_prime]
  s_l, s_lprime = susceptibility[l], susceptibility[l_prime]
  log_grav_term = np.log(d_l_lprime) - np.exp(eta[2]) * (np.log(s_l) + np.log(s_lprime))
  baseline_logit = eta[0] - np.exp(eta[1] + log_grav_term)
  transmission_prob_ = expit(baseline_logit + a[l] * eta[3] + a[l_prime] * eta[4])
  return transmission_prob_


@njit
def get_all_ebola_transmission_probs_njit(a, eta, L, distance_matrix, susceptibility, adjacency_matrix):
  transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    for lprime in range(L):
      if adjacency_matrix[l, lprime]:
        d_l_lprime = distance_matrix[l, lprime]
        s_l, s_lprime = susceptibility[l], susceptibility[lprime]
        log_grav_term = np.log(d_l_lprime) - np.exp(eta[2]) * (np.log(s_l) + np.log(s_lprime))
        baseline_logit = eta[0] - np.exp(eta[1] + log_grav_term)
        transmission_prob_ = expit2(baseline_logit + a[l] * eta[3] + a[lprime] * eta[4])
        transmission_probs_matrix[l, lprime] = transmission_prob_
  return transmission_probs_matrix


@njit
def get_all_continuous_tranmission_probs_njit(a, eta, L, distance_matrix, x, adjacency_matrix):
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
  transmission_probs_matrix = np.zeros((L, L))
  distance_matrix, z = kwargs['distance_matrix'], kwargs['z']
  theta_0 = eta[0]
  theta_1 = eta[1:5]
  theta_2 = eta[5:9]
  theta_3 = eta[9]
  theta_4 = eta[10]
  theta_5 = eta[11]
  theta_6 = eta[12]

  for l in range(L):
    for lprime in range(L):
      if adjacency_matrix[l, lprime]:
        d_l_lprime = distance_matrix[l, lprime]
        x_l = x[l, :]
        x_lprime = x[lprime, :]
        logit = theta_0 + np.dot(theta_1, x_l) + np.dot(theta_2, x_lprime) - theta_3*a[l] - theta_4*a[lprime] - \
          theta_5 * d_l_lprime / np.power(z[l] * z[lprime], theta_6)
        transmission_probs_matrix[l, lprime] = expit2(logit)
  return transmission_probs_matrix


def get_all_ebola_transmission_probs(a, eta, L, **kwargs):
  distance_matrix, susceptibility, adjacency_matrix = \
    kwargs['distance_matrix'], kwargs['susceptibility'], kwargs['adjacency_matrix']
  # transmission_probs_matrix = np.zeros((L, L))
  # for l in range(L):
  #   for lprime in range(L):
  #     if adjacency_matrix[l, lprime] + adjacency_matrix[lprime, l] > 0:
  #       transmission_probs_matrix[l, lprime] = \
  #         transmission_prob(a, l, lprime, eta, distance_matrix, susceptibility)
  transmission_probs_matrix = get_all_ebola_transmission_probs_njit(a, eta, L, distance_matrix, susceptibility,
                                                                    adjacency_matrix)
  return transmission_probs_matrix


def infection_prob_at_location(a, l, eta, current_infected, adjacency_list, distance_matrix, susceptibility):
  if current_infected[l]:
    return 1
  else:
    not_transmitted_prob = np.product([1 - transmission_prob(a, l, l_prime, eta, distance_matrix, susceptibility)
                                       for l_prime in adjacency_list[l]])
    inf_prob = 1 - not_transmitted_prob
    return inf_prob
