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


def success_component(eta0, eta1, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N, N_1, N_inf_neighbor, N_00, N_01,
                      N_11, N_10):
  """
  Component of log likelihood corresponding to infected-at-next-step.
  All of the counts (N*) are only over locations that are i) not infected and ii) infected at next step!
  """
  latent_component = N*eta0
  no_treat_component = -(N - N_1)*np.log(1 + np.exp(eta0))
  treat_component = N_1 * (eta1 - np.log(1 + np.exp(eta0p1)))
  neighbor_latent_component = N_inf_neighbor*eta2
  neighbor_treat_component = N_00*eta2 + N_10*eta2p3 + N_11*eta2p3p4 + N_01*eta2p4 - \
                             N_00 * np.log(1 + np.exp(eta2)) - \
                             N_10 * np.log(1 + np.exp(eta2p3)) - \
                             N_11 * np.log(1 + np.exp(eta2p3p4)) - \
                             N_01 * np.log(1 + np.exp(eta2p4))
  return latent_component + no_treat_component + treat_component + neighbor_latent_component + neighbor_treat_component


def failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N, N_1, N_00, N_01, N_11, N_10):
  """
  Component of log likelihood corresponding to not-infected-at-next-step.
  All of the counts (N*) are only over locations that are i) not infected and ii) not infected at next step!
  """
  no_treat_component = -(N - N_1)*np.log(1 + np.exp(eta0))
  treat_component = -N_1 * np.log(1 + np.exp(eta0p1))
  neighbor_treat_component = -N_00 * np.log(1 + np.exp(eta2)) - \
                              N_10 * np.log(1 + np.exp(eta2p3)) - \
                              N_11 * np.log(1 + np.exp(eta2p3p4)) - \
                              N_01 * np.log(1 + np.exp(eta2p4))
  return no_treat_component + treat_component + neighbor_treat_component


def negative_log_likelihood(env, eta):
  eta0 = eta[0]
  eta1 = eta[1]
  eta0p1 = eta0 + eta1
  eta2 = eta[2]
  eta2p3 = eta2 + eta[3]
  eta2p3p4 = eta2p3 + eta[4]
  eta2p4 = eta2 + eta[4]

  d_i, d_n = env.counts_for_likelihood_dict['next_infected'], \
    env.counts_for_likelihood_dict['next_not_infected']
  N_i, N_1_i, N_inf_neighbor_i, N_00_i, N_01_i, N_11_i, N_10_i = d_i['N'], d_i['N_1'], d_i['N_inf_neighbor'], \
    d_i['N_00'], d_i['N_01'], d_i['N_11'], d_i['N_10']
  N_n, N_1_n, N_inf_neighbor_n, N_00_n, N_01_n, N_11_n, N_10_n = d_n['N'], d_n['N_1'], d_n['N_inf_neighbor'], \
    d_n['N_00'], d_n['N_01'], d_n['N_11'], d_n['N_10']

  lik_success_component = success_component(eta0, eta1, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_i, N_1_i,
                                            N_inf_neighbor_i, N_00_i, N_01_i, N_11_i, N_10_i)
  lik_failure_component = failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, N_n, N_1_n, N_00_n, N_01_n,
                                            N_11_n, N_10_n)
  return -lik_success_component - lik_failure_component

