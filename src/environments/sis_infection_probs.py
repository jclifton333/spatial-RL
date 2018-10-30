import pdb
import numpy as np
from numba import njit, jit

""""
For computing sis model infection probabilities.  See draft pg. 13.
"""


@jit
def expit(logit_p):
  return 1 - 1 / (1 + np.exp(logit_p))


def sis_infection_probability(a, y, eta, L, adjacency_lists, **kwargs):
  """

  :param a:
  :param y:
  :param eta: Transition probability parameter
  :param L: Number of locations
  :param adjacency_lists: List of adjacency lists
  :param kwargs: dict containing value of omega and s

  :return:
  """
  omega, s = kwargs['omega'], kwargs['s']
  z = np.random.binomial(1, omega)
  indicator = (z * s <= 0)
  a_times_indicator = np.multiply(a, indicator)

  infected_indices = np.where(y > 0)[0].astype(int)
  not_infected_indices = np.where(y == 0)[0].astype(int)

  infected_probabilities = np.zeros(L)
  infected_probabilities[not_infected_indices] = p_l(a, a_times_indicator, not_infected_indices, infected_indices, eta,
                                                     adjacency_lists, omega)
  infected_probabilities[infected_indices] = 1 - q_l(a[infected_indices], a_times_indicator[infected_indices], eta, omega)
  return infected_probabilities


def p_l0(a, a_times_indicator, eta, omega):
  logit_p_0 = eta[0] + eta[1] * (a_times_indicator - omega * a)
  p_0 = expit(logit_p_0)
  return p_0


def q_l(a, a_times_indicator, eta, omega):
  logit_q = eta[5] + eta[6] * (a_times_indicator - omega * a)
  q = expit(logit_q)
  return q


def sis_transmission_probs_for_omega0(a, l, lprime, eta, adjacency_matrix):
  """
  p_llprime when omega=0

  :param a:
  :param l:
  :param lprime:
  :param eta:
  :param adjacency_matrix:
  :return:
  """
  if adjacency_matrix[l, lprime] + adjacency_matrix[lprime, l] > 0:
    logit_p_llprime = eta[2] + eta[3]*a[l] + eta[4]*a[lprime]
    p_llprime = expit(logit_p_llprime)
  else:
    p_llprime = 0.0
  return p_llprime


def get_all_sis_transmission_probs_omega0(a, eta, L, adjacency_matrix):
  transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    for lprime in range(L):
      transmission_probs_matrix[l, lprime] = sis_transmission_probs_for_omega0(a, l, lprime, eta, adjacency_matrix)
  return transmission_probs_matrix


@jit
def one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega):
  """

  :param a_times_indicator:
  :param not_infected_indices:
  :param infected_indices:
  :param eta:
  :param adjacency_lists:
  :return:
  """

  product_vector = []
  for l in not_infected_indices:
    neighbors = adjacency_lists[l]
    product_l = 1.0
    for lprime in neighbors:
      if lprime in infected_indices:
        logit_p_llprime = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*(a_times_indicator[lprime] - omega * a[lprime])
        product_l *= 1 - expit(logit_p_llprime)
    product_vector.append(product_l)

  # for l in not_infected_indices[0].tolist():
  #   # Get infected neighbors
  #   infected_neighbor_indices = np.intersect1d(adjacency_lists[l], infected_indices)
  #   a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
  #   logit_p_l = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator_lprime
  #   p_l = expit(logit_p_l)
  #   product_l = np.product(1 - p_l)
  #   product_vector = np.append(product_vector, product_l)

  return product_vector


def p_l(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega):
  p_l0_ = p_l0(a[not_infected_indices], a_times_indicator[not_infected_indices], eta, omega)
  one_minus_p_llprime_ = one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta,
                                             adjacency_lists, omega)
  product = np.multiply(1 - p_l0_, one_minus_p_llprime_)
  return 1 - product
