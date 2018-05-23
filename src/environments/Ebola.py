# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:18:41 2018

@author: Jesse
"""

import numpy as np
from scipy.special import expit
from .SpatialDisease import SpatialDisease
import pickle as pkl
import os
import pdb

class Ebola(SpatialDisease):
  #Load network information
  this_file_pathname = os.path.dirname(os.path.abspath(__file__))
  ebola_network_data_fpath = os.path.join(this_file_pathname, 'ebola-network-data', 'ebola_network_data.p')
  network_info = pkl.load(open(ebola_network_data_fpath, 'rb'))
  ADJACENCY_MATRIX = network_info['adjacency_matrix']
  DISTANCE_MATRIX  = network_info['distance_matrix']
  SUSCEPTIBILITY  = network_info['pop_array']
  L = len(SUSCEPTIBILITY)

  #Placeholder params until actual model is implemented  
  ETA_0 = 1
  ETA_1 = 1
  ETA_2 = 1
  ETA_3 = 1
  ETA_4 = 1
  
  #Compute transmission probs  
  TRANSMISSION_PROBS = np.zeros((L, L, 2, 2))
  for l in range(L):
    s_l = SUSCEPTIBILITY[l]
    for l_prime in range(L):
      d_l_lprime = DISTANCE_MATRIX[l, l_prime]
      s_l_prime = SUSCEPTIBILITY[l_prime]
      baseline_logit = ETA_0 - np.exp(ETA_1) * \
                       (d_l_lprime) / ((s_l * s_l_prime))**ETA_2
      TRANSMISSION_PROBS[l, l_prime, 0, 0] = expit(baseline_logit)
      TRANSMISSION_PROBS[l, l_prime, 1, 0] = expit(baseline_logit + ETA_3)
      TRANSMISSION_PROBS[l, l_prime, 0, 1] = expit(baseline_logit + ETA_4)
      TRANSMISSION_PROBS[l, l_prime, 1, 1] = expit(baseline_logit + ETA_3 + ETA_4)  
      
  INITIAL_INFECT_PROB = np.ones(L) / L
  
  def __init__(self, featureFunction):
    SpatialDisease.__init__(self, Ebola.ADJACENCY_MATRIX, featureFunction)
    
  def reset(self):
    super.reset()
      
  def transmissionProb(self, a, l, l_prime):
    """
    :param a: L-length binary array of treatment decisions
    :param l: index of transmitting location
    :param l_prime: index of transmitted-to location
    """
    if self.current_infected[l]:
      return Ebola.TRANSMISSION_PROBS[l, l_prime, int(a[l]), int(a[l_prime])]
    else:
      return 0
    
  def infectionProb(self, a, l):
    not_infected_prob = np.product([1-self.transmissionProb(a, l_prime, l) for l_prime in self.adjacency_list[l]])
    return 1 - not_infected_prob  

  def updateObsHistory(self, a):
    super.updateObsHistory()
    raw_data_block = np.column_stack((Ebola.SUSCEPTIBILITY, a, self.Y[-2,:]))
    data_block = self.featureFunction(raw_data_block)
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)
    
  def next_infections(self, a):
    super.next_infections(a)
    next_infected_probabilities = np.array([self.infectionProb(a, l) for l in range(self.L)])
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)    
    self.Y = np.vstack((self.Y, next_infections))
    self.R = np.append(self.R, np.sum(next_infections))
    self.true_infection_probs = np.vstack((self.true_infection_probs, next_infected_probabilities))
    self.current_infected = next_infections

  def data_block_at_action(self, data_block, action):
    """
    Replace action in raw data_block with given action.
    """
    super.data_block_at_action(data_block, action)
    assert data_block.shape[1] == 3
    new_data_block = np.column_stack((data_block[:, 0], action, data_block[:, 2]))
    new_data_block = self.featureFunction(new_data_block)
    return new_data_block
    


