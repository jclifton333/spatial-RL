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
  for i in range(6):
    prod = prod * exp_list[i]**powers[i]
  return prod


@njit
def success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_10, N_11):
  """
  Negative log lik for a single success observation.
  """
  prod = exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_10, N_11)
  return np.log(1 - prod)


@njit
def failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_10, N_11):
  prod = exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_0, N_1, N_00, N_01, N_10, N_11)
  return np.log(prod)


@njit
def success_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, success_weights):
  """
  Component of log likelihood corresponding to infected-at-next-step.
  """
  lik = 0
  for i in range(success_weights.shape[1]):
    for j in range(success_weights.shape[2]):
      weight_0_ij = success_weights[0,i,j]
      weight_1_ij = success_weights[1,i,j]
      lik = lik + weight_0_ij*success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 1, 0, i, j, 0, 0) + \
        weight_1_ij*success_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 0, 1, 0, 0, j, i)
  return lik


@njit
def failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00, n_01, n_10, n_11):
  """
  Component of log likelihood corresponding to not-infected-at-next-step.
  """
  for i in range(failure_weights.shape[1]):
    for j in range(failure_weights.shape[2]):
      weight_0_ij = failure_weights[0,i,j]
      weight_1_ij = failure_weights[1,i,j]
      lik = lik + weight_0_ij*failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 1, 0, i, j, 0, 0) + \
        weight_1_ij*failure_component_single(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, 0, 1, 0, 0, j, i)
  return lik


# ToDo: jitify!
def get_bootstrap_weights(all_weights, counts_for_likelihood, indices_for_likelihood):
  weights = np.zeros(shape=counts_for_likelihood.shape)
  for i in range(counts_for_likelihood.shape[0]):
    for j in range(counts_for_likelihood.shape[1]):
      for k in range(counts_for_likelihood.shape[2]):
        indices = indices_for_likelihood[i][j][k]
        weights[i,j,k] += np.sum([all_weights[ix] for ix in indices])
  return weights


def negative_log_likelihood(eta, counts_for_likliehood):
  """

  :param eta:
  :param counts_for_likelihood:
  :return:
  """
  eta0 = eta[0]
  eta0p1 = eta0 + eta[1]
  eta2 = eta[2]
  eta2p3 = eta2 + eta[3]
  eta2p3p4 = eta2p3 + eta[4]
  eta2p4 = eta2 + eta[4]

  n_00_1, n_01_1, n_10_1, n_11_1 = counts_for_likliehood['n_00_1'], counts_for_likliehood['n_01_1'], \
                                   counts_for_likliehood['n_10_1'], counts_for_likliehood['n_11_1']
  n_00_0, n_01_0, n_10_0, n_11_0 = ccounts_for_likliehood['n_00_0'], counts_for_likliehood['n_01_0'], \
                                   counts_for_likliehood['n_10_0'], counts_for_likliehood['n_11_0']

  lik_success_component = \
    success_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00_1, n_01_1, n_10_1, n_11_1)
  lik_failure_component = \
    failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00_0, n_01_0, n_10_0, n_11_0)
  return -lik_success_component - lik_failure_component

