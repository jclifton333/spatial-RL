import pdb
import numpy as np
from numba import njit, jit

""""
For computing sis model infection probabilities.  See draft pg. 13.
"""


@jit
def expit(logit_p):
  return 1 - 1 / (1 + np.exp(logit_p))


def sis_infection_probability(a, y, s, eta, omega, L, adjacency_lists):
  """

  :param a:
  :param y:
  :param s:
  :param eta: Transition probability parameter
  :param omega: shield state mixing parameter
  :param L: Number of locations
  :param adjacency_lists: List of adjacency lists
  :return:
  """
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
