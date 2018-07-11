"""
Functions for specifying the component of the log likelihood corresponding to p (infection probabilities
for non-infected states).

Variables N* count the number of different infection/treatment/neighbor infection/treatment combinations in the dataset;
these are computed online and are attributes of env.

The subscript of N_ is an indicator of treatment: N_1 is the number of treated locations, N_01 is the number of
untreated-treated location-neighbor pairs, etc.  And sum_ij N_ij = N_inf_neighbor.

Finally, _i stands for "infected", while _n stands for "not infected", at next step.
"""
import numpy as np
from numba import njit
import math


@njit
def exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  """
  Helper for computing likelihood.
  """
  exp_0 = 1 + np.exp(eta0 + a*eta1)
  exp_00 = 1 + np.exp(eta2)
  exp_01 = 1 + np.exp(eta2p4)
  exp_10 = 1 + np.exp(eta2p3)
  exp_11 = 1 + np.exp(eta2p3p4)
  exp_list = [exp_0, exp_00, exp_01, exp_10, exp_11]
  powers = [-1, -N_00, -N_01, -N_10, -N_11]
  prod = 1
  for i in range(5):
    prod = prod * exp_list[i]**powers[i]
  return prod


def success_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  """
  Negative log lik for a single success observation.
  """
  prod = exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return -math.log(1 - prod)


def failure_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  prod = exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return math.log(prod)


@njit
def component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, success_likelihood_counts_list):
  """
  Component of log likelihood corresponding to infected-at-next-step.
  All of the counts (N*) are only over locations that are i) not infected and ii) infected at next step!
  """
  lik = 0
  for counts in success_likelihood_counts_list:
    a = counts[1]
    N = counts[0]
    N_00, N_01, N_10, N_11 = N[0,0], N[0,1], N[1,0], N[1,1]
    lik = lik + success_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return lik


def negative_log_likelihood(eta, env):
  eta0 = eta[0]
  eta1 = eta[1]
  eta2 = eta[2]
  eta2p3 = eta2 + eta[3]
  eta2p3p4 = eta2p3 + eta[4]
  eta2p4 = eta2 + eta[4]

  lik_success_component = component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, env.counts_for_likelihood_next_infected)
  lik_failure_component = component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4,
                                    env.counts_for_likelihood_next_not_infected)
  return -lik_success_component - lik_failure_component

