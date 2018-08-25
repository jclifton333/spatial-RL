"""
Component of log likelihood for infection probabilities at not-infected states ("p_l" in the draft).
"""

import pdb
import numpy as np
from numba import njit


@njit
def exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00, n_01, n_10, n_11):
  """
  Component of log likelihood corresponding to not-infected-at-next-step.
  """
  exp_0 = 1 + np.exp(eta0)
  exp_1 = 1 + np.exp(eta0p1)
  exp_00 = 1 + np.exp(eta2)
  exp_01 = 1 + np.exp(eta2p4)
  exp_10 = 1 + np.exp(eta2p3)
  exp_11 = 1 + np.exp(eta2p3p4)

  prod = 1
  for n_00_, n_01_, n_10_, n_11_ in zip(n_00, n_01, n_10, n_11):
    prod *= np.power(exp_0, -(n_00_ + n_01_ > 0)) * np.power(exp_1, -(n_10_ + n_11_ > 0)) * np.power(exp_00, -n_00_) * \
      np.power(exp_01, -n_01_) * np.power(exp_10, -n_10_) * np.power(exp_11, -n_11_)
  return prod


def success_or_failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00, n_01, n_10, n_11,
                                 success):
  exp_prod_ = exp_prod(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, eta2p4, n_00, n_01, n_10, n_11)
  if success:
    return np.log(1 - exp_prod_)
  else:
    return np.log(exp_prod_)


def negative_log_likelihood(eta, counts_for_likelihood):
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

  n_00_1, n_01_1, n_10_1, n_11_1 = counts_for_likelihood['n_00_1'], counts_for_likelihood['n_01_1'], \
                                   counts_for_likelihood['n_10_1'], counts_for_likelihood['n_11_1']
  n_00_0, n_01_0, n_10_0, n_11_0 = counts_for_likelihood['n_00_0'], counts_for_likelihood['n_01_0'], \
                                   counts_for_likelihood['n_10_0'], counts_for_likelihood['n_11_0']

  lik_success_component = \
    success_or_failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00_1, n_01_1, n_10_1, n_11_1,
                                 success=1)
  lik_failure_component = \
    success_or_failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, n_00_0, n_01_0, n_10_0, n_11_0,
                                 success=0)
  return -lik_success_component - lik_failure_component

