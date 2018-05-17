# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:18:41 2018

@author: Jesse
"""

import numpy as np
from scipy.special import expit

class Ebola(object):
  #Placeholder params until actual model is implemented
  L = 9
  
  ETA_0 = 1
  ETA_1 = 1
  ETA_2 = 1
  ETA_3 = 1
  ETA_4 = 1
  
  ADJACENCY_MATRIX = np.random.binomial(1, 0.3, size=(L, L))
  ADJACENCY_LIST = [[lprime for lprime in range(L) if ADJACENCY_MATRIX[l, lprime] == 1] for l in range(L)]
  DISTANCE_MATRIX  = np.random.uniform(high=1000, size=(L,L))
  SUSCEPTIBILITY = np.random.normal(size=L)
  
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
  
  def __init__(self):
    self.currentInfected = np.zeros(L)
  
  def transmissionProb(self, a, l, l_prime):
    '''
    :param a: L-length binary array of treatment decisions
    :param l: index of transmitting location
    :param l_prime: index of transmitted-to location
    '''
    if self.currentInfected[l]:
      return Ebola.TRANSMISSION_PROBS[l, l_prime, a[l], a[l_prime]]
    else:
      return 0
    
  def infectionProb(self, a, l):
    not_infected_prob = np.product([1-self.transmissionProb(a, l, l_prime) for l_prime in Ebola.ADJACENCY_LIST[l]])
    return 1 - not_infected_prob
  
  def step(self, a):
    infectionProbs = np.array([self.infectionProb(a, l) for l in range(Ebola.L)])
    nextInfected = np.random.binomial(n=[1]*Ebola.L, p=infectionProbs)
    
    
    
    
    
  
  