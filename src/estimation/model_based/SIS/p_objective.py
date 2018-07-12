import pdb
import numpy as np
from numba import njit, jit


@njit
def exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  """
  Helper for computing likelihood.

  ToDo: If necessary, this can be computed more efficiently by tracking the number of each N_* combination. There are
        2(N+1) such combos where N = max number of neighbors in network.
  """
  exp_0 = 1 + np.exp(eta0 + a*eta1)
  exp_00 = 1 + np.exp(eta2)
  exp_01 = 1 + np.exp(eta2p4)
  exp_10 = 1 + np.exp(eta2p3)
  exp_11 = 1 + np.exp(eta2p3p4)
  exp_list = np.array([exp_0, exp_00, exp_01, exp_10, exp_11])
  powers = np.array([-1, -N_00, -N_01, -N_10, -N_11])
  prod = 1
  for i in range(5):
    prod = prod * exp_list[i]**powers[i]
  return prod


@njit
def success_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  """
  Negative log lik for a single success observation.
  """
  prod = exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return np.log(1 - prod)


@njit
def failure_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10):
  prod = exp_prod(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return -np.log(prod)


@njit
def component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, success_likelihood_counts_list,
              success_likelihood_actions_list):
  """
  Component of log likelihood corresponding to infected-at-next-step.
  All of the counts (N*) are only over locations that are i) not infected and ii) infected at next step!
  """
  lik = 0
  for i in range(success_likelihood_counts_list.shape[0]):
    N_00 = success_likelihood_counts_list[i,0,0]
    N_10 = success_likelihood_counts_list[i,1,0]
    N_11 = success_likelihood_counts_list[i,1,1]
    N_01 = success_likelihood_counts_list[i,0,1]
    a = success_likelihood_actions_list[i]
    lik = lik + success_component_single(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4, a, N_00, N_01, N_11, N_10)
  return lik


def negative_log_likelihood(eta, env):
  eta = -eta
  eta0 = eta[0]
  eta1 = eta[1]
  eta2 = eta[2]
  eta2p3 = eta2 + eta[3]
  eta2p3p4 = eta2p3 + eta[4]
  eta2p4 = eta2 + eta[4]

  lik_success_component = component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4,
                                    np.array(env.counts_for_likelihood_next_infected),
                                    np.array(env.actions_for_likelihood_next_infected))
  lik_failure_component = component(eta0, eta1, eta2, eta2p3, eta2p3p4, eta2p4,
                                    np.array(env.counts_for_likelihood_next_not_infected),
                                    np.array(env.actions_for_likelihood_next_not_infected))
  return -lik_success_component - lik_failure_component

