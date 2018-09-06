# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:18:41 2018

@author: Jesse
"""

import numpy as np
import copy
from scipy.special import expit
from src.environments.SpatialDisease import SpatialDisease
from src.environments.sis import SIS
import pickle as pkl
import os
import pdb


class Ebola(SpatialDisease):
  # Load network information
  this_file_pathname = os.path.dirname(os.path.abspath(__file__))
  ebola_network_data_fpath = os.path.join(this_file_pathname, 'ebola-network-data', 'ebola_network_data.p')
  network_info = pkl.load(open(ebola_network_data_fpath, 'rb'))
  # ADJACENCY_MATRIX = network_info['adjacency_matrix']

  # DISTANCE_MATRIX  = network_info['haversine_distance_matrix']
  DISTANCE_MATRIX = network_info['euclidean_distance_matrix']
  SUSCEPTIBILITY = network_info['pop_array']
  L = len(SUSCEPTIBILITY)
  OUTBREAK_TIMES = network_info['outbreak_time_array']

  ADJACENCY_MATRIX = np.zeros((L, L))
  for l in range(L):
    d_l_lprime = DISTANCE_MATRIX[l, :]
    s_l_lprime = SUSCEPTIBILITY[l] * SUSCEPTIBILITY
    sorted_ratios = np.argsort(d_l_lprime / s_l_lprime)
    for lprime in sorted_ratios[1:4]:
        ADJACENCY_MATRIX[l, lprime] = 1

  MAX_NUMBER_OF_NEIGHBORS = int(np.max(np.sum(ADJACENCY_MATRIX, axis=1)))

  # Fill matrix of susceptibility products
  PRODUCT_MATRIX = np.outer(SUSCEPTIBILITY, SUSCEPTIBILITY)

  # Get initial outbreaks
  OUTBREAK_TIMES[np.where(OUTBREAK_TIMES == -1)] = np.max(OUTBREAK_TIMES) + 1 # Make it easier to sort
  NUMBER_OF_INITIAL_OUTBREAKS = 25
  OUTBREAK_INDICES = np.argsort(OUTBREAK_TIMES)[:NUMBER_OF_INITIAL_OUTBREAKS]
  INITIAL_INFECTIONS = np.zeros(L)
  INITIAL_INFECTIONS[OUTBREAK_INDICES] = 1

  # Params for logit of transmission probability
  ETA_0 = SIS.ETA_2
  ETA_1 = SIS.ETA_3
  ETA_2 = 0.0
  ETA_3 = SIS.ETA_3
  ETA_4 = SIS.ETA_4

  # ALPHA = 1.24
  # ETA_0 = -8 * ALPHA
  # ETA_1 = np.log(156) + np.log(ALPHA)
  # ETA_2 = 5
  # ETA_3 = -8.0
  # ETA_4 = -8.0
  # ETA_0 = -7.2 * ALPHA
  # ETA_1 = -0.284 + np.log(ALPHA)
  # ETA_2 = -0.0
  # ETA_3 = -1.015 
  # ETA_4 = -1.015
  ETA = np.array([ETA_0, np.exp(ETA_1), np.exp(ETA_2), ETA_3, ETA_4])
  # Compute transmission probs
  TRANSMISSION_PROBS = np.zeros((L, L, 2, 2))
  for l in range(L):
    s_l = SUSCEPTIBILITY[l]
    for l_prime in range(L):
      # if ADJACENCY_MATRIX[l, l_prime] == 1 or ADJACENCY_MATRIX[l_prime, l] == 1:
      if True:
        """
        from https://github.com/LaberLabs/stdmMf_cpp/blob/master/src/main/ebolaStateGravityModel.cpp
        
        log_grav_term = log(dist(a, b)) - exp(beta_2)*( log(pop(a)) + log(pop(b)))
        logit_prob = beta_0 - exp(beta_1)*log_grav_term
        """
        d_l_lprime = DISTANCE_MATRIX[l, l_prime]
        s_l_prime = SUSCEPTIBILITY[l_prime]
        log_grav_term = np.log(d_l_lprime) - np.exp(ETA_2)*(np.log(s_l) + np.log(s_l_prime))
        baseline_logit = ETA_0 - np.exp(ETA_1 + log_grav_term)
        # baseline_logit = ETA_0
        TRANSMISSION_PROBS[l, l_prime, 0, 0] = expit(baseline_logit)
        TRANSMISSION_PROBS[l, l_prime, 1, 0] = expit(baseline_logit + ETA_3)
        TRANSMISSION_PROBS[l, l_prime, 0, 1] = expit(baseline_logit + ETA_4)
        TRANSMISSION_PROBS[l, l_prime, 1, 1] = expit(baseline_logit + ETA_3 + ETA_4)

  def __init__(self, eta=None):
    SpatialDisease.__init__(self, Ebola.ADJACENCY_MATRIX, initial_infections=Ebola.INITIAL_INFECTIONS)

    # Modify eta if one is given
    if eta is not None:
      ETA_0, ETA_1, ETA_2, ETA_3, ETA_4 = eta
      self.TRANSMISSION_PROBS = np.zeros((self.L, self.L, 2, 2))
      for l in range(self.L):
        s_l = Ebola.SUSCEPTIBILITY[l]
        for l_prime in range(self.L):
          # if ADJACENCY_MATRIX[l, l_prime] == 1 or ADJACENCY_MATRIX[l_prime, l] == 1:
          if True:
            d_l_lprime = Ebola.DISTANCE_MATRIX[l, l_prime]
            s_l_prime = Ebola.SUSCEPTIBILITY[l_prime]
            log_grav_term = np.log(d_l_lprime) - np.exp(ETA_2)*(np.log(s_l) + np.log(s_l_prime))
            baseline_logit = ETA_0 - np.exp(ETA_1 + log_grav_term)
            self.TRANSMISSION_PROBS[l, l_prime, 0, 0] = expit(baseline_logit)
            self.TRANSMISSION_PROBS[l, l_prime, 1, 0] = expit(baseline_logit + ETA_3)
            self.TRANSMISSION_PROBS[l, l_prime, 0, 1] = expit(baseline_logit + ETA_4)
            self.TRANSMISSION_PROBS[l, l_prime, 1, 1] = expit(baseline_logit + ETA_3 + ETA_4)

    # Initial steps
    self.step(np.zeros(self.L))
    self.step(np.zeros(self.L))

  def train_test_split(self):
    pass

  def reset(self):
    super(Ebola, self).reset()
    # Initial steps
    self.step(np.zeros(self.L))
    self.step(np.zeros(self.L))

  def transmission_prob(self, a, l, l_prime, eta):
    """
    :param a: L-length binary array of treatment decisions
    :param l_prime: index of transmitting location
    :param l: index of transmitted-to location
    :param eta: transmission prob parameters, or None; if None use self.TRANMISSION_PROBS
    """
    if self.current_infected[l_prime]:
      if eta is None:
        transmission_prob = self.TRANSMISSION_PROBS[l, l_prime, int(a[l]), int(a[l_prime])]
      else:
        d_l_lprime = self.DISTANCE_MATRIX[l, l_prime]
        s_l, s_lprime = self.SUSCEPTIBILITY[l], self.SUSCEPTIBILITY[l_prime]
        log_grav_term = np.log(d_l_lprime) - np.exp(eta[2])*(np.log(s_l) + np.log(s_lprime))
        baseline_logit = eta[0] - np.exp(eta[1] + log_grav_term)
        transmission_prob = expit(baseline_logit + a[l]*eta[3] + a[l_prime]*a[4])
      return transmission_prob
    else:
      return 0

  def infection_prob_at_location(self, a, l, eta):
    if self.current_infected[l]:
      return 1
    else:
      not_transmitted_prob = np.product([1-self.transmission_prob(a, l, l_prime, eta) for l_prime in self.adjacency_list[l]])
      # not_transmitted_prob = np.product([1-self.transmission_prob(a, l, l_prime, eta) for l_prime in range(self.L)])
      # latent_inf_prob = expit(10*(Ebola.ETA_0 + a[l] * 50*Ebola.ETA_3))
      # inf_prob = 1 - ((1 - latent_inf_prob) * not_transmitted_prob)
      inf_prob = 1 - not_transmitted_prob
      return inf_prob

  def next_infected_probabilities(self, a, eta=None):
    return np.array([self.infection_prob_at_location(a, l, eta) for l in range(self.L)])

  def update_obs_history(self, a):
    super(Ebola, self).update_obs_history(a)
    raw_data_block = np.column_stack((Ebola.SUSCEPTIBILITY, a, self.Y[-2,:]))
    data_block = self.psi(raw_data_block)
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)

  def next_state(self):
    super(Ebola, self).next_state()

  def next_infections(self, a):
    super(Ebola, self).next_infections(a)
    next_infected_probabilities = self.next_infected_probabilities(a)
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.true_infection_probs.append(next_infected_probabilities)
    self.current_infected = next_infections

  def data_block_at_action(self, data_block_ix, action, raw=False):
    """
    Replace action in raw data_block with given action.
    """
    super(Ebola, self).data_block_at_action(data_block_ix, action)
    if raw:
      new_data_block = copy.copy(self.X_raw[data_block_ix])
      new_data_block[:, 1] = action
    else:
      new_data_block = self.psi_at_action(self.X[data_block_ix], self.A[data_block_ix, :], action)
    return new_data_block

  # Neighbor features #
  def psi_at_location(self, l, raw_data_block):
    x_l = raw_data_block[l, :]
    neighbors = self.adjacency_list[l]
    num_neighbors = len(neighbors)
    for i in range(self.MAX_NUMBER_OF_NEIGHBORS):
      if i >= num_neighbors:  # zero padding
        x_l = np.concatenate((x_l, np.zeros(4)))
      else:
        l_prime = neighbors[i]
        x_lprime = raw_data_block[l_prime, :]
        x_l = np.concatenate((x_l, x_lprime, [self.DISTANCE_MATRIX[l, l_prime] + self.DISTANCE_MATRIX[l_prime, l]]))
    return x_l

  def psi(self, raw_data_block):
    # (s, a, y) for location and (s, a, y, d) for its neighbors
    number_of_features = int(3 + 4*self.MAX_NUMBER_OF_NEIGHBORS)
    X = np.zeros((0, number_of_features))
    for l in range(self.L):
      x_l = self.psi_at_location(l, raw_data_block)
      X = np.vstack((X, x_l))
    return X

  def psi_at_action(self, old_data_block, old_action, action):
    new_data_block = copy.copy(old_data_block)
    locations_with_changed_actions = set(np.where(old_action != action)[0])

    for l in range(self.L):
      # Check if action at l changed
      if l in locations_with_changed_actions:
        new_data_block[l, 1] = action[l]

      # Check if actions at l neighbors have changed
      for i in range(len(self.adjacency_list[l])):
        l_prime = self.adjacency_list[l][i]
        if l_prime in locations_with_changed_actions:
          new_data_block[l, 3 + i*4 + 1] = action[l_prime]
    return new_data_block

    


