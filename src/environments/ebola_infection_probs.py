import numpy as np
from scipy.special import expit


def ebola_infection_probs(a, y, eta, L, adjacency_lists, **kwargs):
  distance_matrix, susceptibility = kwargs['distance_matrix'], kwargs['susceptibility']
  return np.array([infection_prob_at_location(a, l, eta, y, adjacency_lists, distance_matrix,
                                              susceptibility) for l in range(L)])


def transmission_prob(a, l, l_prime, eta, distance_matrix, susceptibility):
  d_l_lprime = distance_matrix[l, l_prime]
  s_l, s_lprime = susceptibility[l], susceptibility[l_prime]
  log_grav_term = np.log(d_l_lprime) - np.exp(eta[2]) * (np.log(s_l) + np.log(s_lprime))
  baseline_logit = eta[0] - np.exp(eta[1] + log_grav_term)
  transmission_prob_ = expit(baseline_logit + a[l] * eta[3] + a[l_prime] * a[4])
  return transmission_prob_


def get_all_ebola_transmission_probs(a, eta, L, **kwargs):
  distance_matrix, susceptibility = kwargs['distance_matrix'], kwargs['susceptibility']
  transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    for lprime in range(L):
      transmission_probs_matrix[l, lprime] = transmission_prob(a, l, lprime, eta, distance_matrix, susceptibility)
  return transmission_probs_matrix


def infection_prob_at_location(a, l, eta, current_infected, adjacency_list, distance_matrix, susceptibility):
  if current_infected[l]:
    return 1
  else:
    not_transmitted_prob = np.product([1 - transmission_prob(a, l, l_prime, eta, distance_matrix, susceptibility)
                                       for l_prime in adjacency_list[l]])
    inf_prob = 1 - not_transmitted_prob
    return inf_prob