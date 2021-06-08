import pdb
import numpy as np
from numba import njit, jit
import math
import copy

""""
For computing sis model infection probabilities.  See draft pg. 13.
"""


@njit
def expit(logit_p):
  return 1 - 1 / (1 + np.exp(logit_p))


def sis_infection_probability_oracle_contaminated(a, y, eta, L, adjacency_lists, epsilon, contaminator, feature_function,
                                                  **kwargs):
  if epsilon > 0:
    s = kwargs['s']
    s_indicator = s > 0
    X_raw_ = np.column_stack((s_indicator, a, y))
    X_ = feature_function(X_raw_, neighbor_order=1)
    contaminator_probs = contaminator.predict_proba(X_)
    if epsilon < 1:
      base_infected_probabilities = sis_infection_probability(a, y, eta, L, adjacency_lists, **kwargs)
      probs = (1 - epsilon) * base_infected_probabilities + epsilon * contaminator_probs
      return probs
    else:
      return contaminator_probs
  else:
    base_infected_probabilities = sis_infection_probability(a, y, eta, L, adjacency_lists, **kwargs)
    return base_infected_probabilities


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
  # logit_p_0 = eta[0] + eta[1] * (a_times_indicator - omega * a)
  logit_p_0 = eta[0] + eta[1] * a_times_indicator
  p_0 = expit(logit_p_0)
  return p_0


def q_l(a, a_times_indicator, eta, omega):
  # logit_q = eta[5] + eta[6] * (a_times_indicator - omega * a)
  logit_q = eta[5] + eta[6] * a_times_indicator
  q = expit(logit_q)
  return q


# @njit
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


def get_all_oracle_contaminated_sis_transmission_probs(a, eta, L, **kwargs):
  epsilon = kwargs['epsilon']
  if epsilon > 0:
    contaminated = get_all_pseudo_transmission_probs(a, eta, L, **kwargs)
    if epsilon < 1:
      uncontaminated = get_all_sis_transmission_probs_omega0(a, eta, L, **kwargs)
      final_transmission_probs = epsilon * uncontaminated + (1 - epsilon) * contaminated
      return final_transmission_probs
    else:
      return contaminated
  else:
    return get_all_sis_transmission_probs_omega0(a, eta, L, **kwargs)


def get_all_pseudo_transmission_probs(a, eta, L, **kwargs):
  s, y, feature_function, contaminator, adjacency_matrix = \
    kwargs['s'], kwargs['y'], kwargs['feature_function'], kwargs['contaminator'], kwargs['adjacency_matrix']
  s_indicator = s > 0
  X_encodings = (s_indicator + 2*a + 4*y).astype(int)
  X_encodings_onehot = np.zeros((L, 8))
  X_raw_ = np.column_stack((s_indicator, a, y))
  X_ = feature_function(X_raw_, neighbor_order=1)
  for l in range(L):
    X_encodings_onehot[l, X_encodings[l]] = 1
  logits = contaminator.get_logit(X_)
  neighbor_logits = contaminator.get_neighbor_logit(X_encodings_onehot)
  exp_logits = np.exp(logits)
  exp_neighbor_logits = (1 + np.exp(neighbor_logits))
  pseudo_transmission_probs_matrix = \
    get_all_pseudo_transmission_probs_wrapped(L, exp_logits, exp_neighbor_logits, adjacency_matrix)
  return pseudo_transmission_probs_matrix


@njit
def get_all_pseudo_transmission_probs_wrapped(L, exp_logits, exp_neighbor_logits, adjacency_matrix):
  pseudo_transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    exp_logits_l = exp_logits[l]
    for lprime in range(L):
      if adjacency_matrix[l, lprime] + adjacency_matrix[lprime, l] > 0:
        exp_neighbor_logits_lprime = exp_neighbor_logits[lprime]
        pseudo_transmission_probs_matrix[l, lprime] = exp_neighbor_logits_lprime / exp_logits_l
  return pseudo_transmission_probs_matrix


def get_all_sis_transmission_probs_omega0(a, eta, L, **kwargs):
  adjacency_matrix = kwargs['adjacency_matrix']
  eta_nb_compatible = [eta_ for eta_ in eta]
  return get_all_sis_transmission_probs_omega0_wrapped(a, eta_nb_compatible, L, adjacency_matrix)


@njit
def get_all_sis_transmission_probs_omega0_wrapped(a, eta_nb_compatible, L, adjacency_matrix):
  transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    for lprime in range(L):
      if adjacency_matrix[l, lprime] + adjacency_matrix[lprime, l] > 0:
        logit_p_llprime = eta_nb_compatible[2] + eta_nb_compatible[3]*a[l] + eta_nb_compatible[4]*a[lprime]
        p_llprime = expit(logit_p_llprime)
      else:
        p_llprime = 0.0
      transmission_probs_matrix[l, lprime] = p_llprime
  return transmission_probs_matrix


@jit
def one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega,
                        len_not_infected, product_vector):
  """

  :param a_times_indicator:
  :param not_infected_indices:
  :param infected_indices:
  :param eta:
  :param adjacency_lists:
  :return:
  """
  product_vector = []
  # for l in not_infected_indices:
  #   product_l = 1.0
  #   neighbors = adjacency_lists[l]
  #   for lprime in neighbors:
  #     if lprime in infected_indices:
  #       # logit_p_llprime = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*(a_times_indicator[lprime] - omega * a[lprime])
  #       logit_p_llprime = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*(a_times_indicator[lprime])
  #       product_l *= 1 - expit(logit_p_llprime)
  #   product_vector.append(product_l)

  for l in not_infected_indices.tolist():
    # Get infected neighbors
    infected_neighbor_indices = np.intersect1d(adjacency_lists[l], infected_indices)
    a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
    logit_p_l = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator_lprime
    p_l = expit(logit_p_l)
    product_l = np.product(1 - p_l)
    product_vector = np.append(product_vector, product_l)

  return product_vector


def p_l(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega):
  p_l0_ = p_l0(a[not_infected_indices], a_times_indicator[not_infected_indices], eta, omega)
  len_not_infected = len(not_infected_indices)
  initial_product_vector = np.ones(len_not_infected)
  one_minus_p_llprime_ = one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta,
                                             adjacency_lists, omega, len_not_infected, initial_product_vector)
  product = np.multiply(1 - p_l0_, one_minus_p_llprime_)
  return 1 - product
