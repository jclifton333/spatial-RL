import pdb
import numpy as np
from numba import njit, jit

""""
For computing SIS model infection probabilities.  See draft pg. 13.
"""


@njit
def expit(logit_p):
  return 1 - 1 / (1 + np.exp(logit_p))


def infection_probability(a, y, s, eta, omega, L, adjacency_lists):
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
  infected_probabilities[not_infected_indices] = p_l(a_times_indicator, not_infected_indices, infected_indices, eta,
                                                     adjacency_lists)
  infected_probabilities[infected_indices] = 1 - q_l(a_times_indicator[infected_indices], eta)
  return infected_probabilities


def p_l0(a_times_indicator, eta):
  logit_p_0 = eta[0] + eta[1] * a_times_indicator
  p_0 = expit(logit_p_0)
  return p_0


def q_l(a_times_indicator, eta):
  logit_q = eta[5] + eta[6] * a_times_indicator
  q = expit(logit_q)
  return q


@jit
def one_minus_p_llprime(a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists,
                        product_vector):
  """

  :param a_times_indicator:
  :param not_infected_indices:
  :param infected_indices:
  :param eta:
  :param adjacency_lists:
  :param product_vector: array of 1s of length len(not_infected_indices)
  :return:
  """

  i = 0
  for l in not_infected_indices:
    neighbors = adjacency_lists[l]
    product_l = 1.0
    for lprime in neighbors:
      if lprime in infected_indices:
        logit_p_llprime = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator[lprime]
        product_l *= 1 - expit(logit_p_llprime)
    product_vector[i] = product_l
    i += 1

  # for l in not_infected_indices[0].tolist():
  #   # Get infected neighbors
  #   infected_neighbor_indices = np.intersect1d(adjacency_lists[l], infected_indices)
  #   a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
  #   logit_p_l = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator_lprime
  #   p_l = expit(logit_p_l)
  #   product_l = np.product(1 - p_l)
  #   product_vector = np.append(product_vector, product_l)

  return product_vector


def p_l(a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists):
  p_l0_ = p_l0(a_times_indicator[not_infected_indices], eta)
  one_minus_p_llprime_ = one_minus_p_llprime(a_times_indicator, not_infected_indices, infected_indices, eta,
                                             adjacency_lists, np.array([1]*len(not_infected_indices)))
  product = np.multiply(1 - p_l0_, one_minus_p_llprime_)
  return 1 - product