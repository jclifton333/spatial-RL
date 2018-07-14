"""
Component of log likelihood for infection probabilities at not-infected states ("p_l" in the draft).
"""

import pdb
import numpy as np
from numba import njit


@njit
def exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_10, N_11):
  """
  Helper for computing likelihood.
  """
  exp_0 = 1 + np.exp(eta0)
  exp_1 = 1 + np.exp(eta0p1)
  exp_00 = 1 + np.exp(eta2)
  exp_01 = 1 + np.exp(eta2p4)
  exp_10 = 1 + np.exp(eta2p3)
  exp_11 = 1 + np.exp(eta2p3p4)
  exp_list = np.array([exp_0, exp_1, exp_00, exp_01, exp_10, exp_11])
  powers = np.array([-N_0, -N_1, -N_00, -N_01, -N_10, -N_11])
  prod = 1
  for i in range(5):
    prod = prod * exp_list[i]**powers[i]
  return prod


@njit
def success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_11, N_10):
  """
  Negative log lik for a single success observation.
  """
  prod = exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_11, N_10)
  return np.log(1 - prod)


@njit
def failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_11, N_10):
  prod = exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_11, N_10)
  return np.log(prod)


@njit
def success_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, success_likelihood_counts_list):
  """
  Component of log likelihood corresponding to infected-at-next-step.
  """
  lik = 0
  for i in range(success_likelihood_counts_list.shape[1]):
    for j in range(success_likelihood_counts_list.shape[2]):
      num_0_ij = success_likelihood_counts_list[0,i,j]
      num_1_ij = success_likelihood_counts_list[1,i,j]
      lik = lik + num_0_ij*success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 1, 0, i, j, 0, 0) + \
        num_1_ij*success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 0, 1, 0, 0, i, j)
  return lik


@njit
def failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, failure_likelihood_counts_list):
  """
  Component of log likelihood corresponding to not-infected-at-next-step.
  """
  lik = 0
  for i in range(failure_likelihood_counts_list.shape[1]):
    for j in range(failure_likelihood_counts_list.shape[2]):
      num_0_ij = failure_likelihood_counts_list[0,i,j]
      num_1_ij = failure_likelihood_counts_list[1,i,j]
      lik = lik + num_0_ij*failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 1, 0, i, j, 0, 0) + \
        num_1_ij*failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 0, 1, 0, 0, i, j)
  return lik


def negative_log_likelihood(eta, counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected):
  eta0 = eta[0]
  eta0p1 = eta0 + eta[1]
  eta2 = eta[2]
  eta2p3 = eta2 + eta[3]
  eta2p3p4 = eta2p3 + eta[4]
  eta2p4 = eta2 + eta[4]

  lik_success_component = success_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4,
                                            counts_for_likelihood_next_infected)
  lik_failure_component = failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4,
                                            counts_for_likelihood_next_not_infected)
  return -lik_success_component - lik_failure_component

