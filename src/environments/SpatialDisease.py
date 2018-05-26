# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:58:36 2018

@author: Jesse
"""
from abc import ABCMeta, abstractmethod
import numpy as np

# Create ABC base class compatible with Python 2.7 and 3.x
ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class SpatialDisease(ABC):
  INITIAL_INFECT_PROB = 0.2
  
  def __init__(self, adjacency_matrix, featureFunction, initialInfections=None):
    """
    :param adjacency_matrix: 2d binary array corresponding to network for gen model 
    :param featureFunction:
    :param initialInfections: L-length binary array of initial infections, or None
    """
    
    self.featureFunction = featureFunction
    self.initialInfections = initialInfections
    # Generative model parameters
    self.L = adjacency_matrix.shape[0]
    
    # Adjacency info
    self.adjacency_matrix = adjacency_matrix
    self.adjacency_list = [[l_prime for l_prime in range(self.L) if self.adjacency_matrix[l, l_prime] == 1] for l in range(self.L)]
    
    # Observation history
    if self.initialInfections is None:
      self.Y = np.array([np.random.binomial(n=1, p=SpatialDisease.INITIAL_INFECT_PROB, size=self.L)])
    else:
      self.Y = np.array([self.initialInfections])
    self.A = np.zeros((0, self.L))
    self.R = np.array([np.sum(self.Y[-1,:])])
    self.X_raw = [] # Will hold blocks [S_t, A_t, Y_t] at each time t
    self.X = []  # Will hold features of [S_t, A_t, Y_t] each each time t
    self.y = []  # Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = np.zeros((0, self.L))
    
    # Current network status
    self.current_infected = self.Y[-1,:]
    self.T = 0
    
  def reset(self):
    """
    Reset state and observation histories.
    """
    # Observation history
    if self.initialInfections is None:
      self.Y = np.array([np.random.binomial(n=1, p=SpatialDisease.INITIAL_INFECT_PROB, size=self.L)])
    else:
      self.Y = np.array([self.initialInfections])
    self.A = np.zeros((0, self.L))
    self.R = np.array([np.sum(self.Y[-1,:])])
    self.X_raw = []
    self.X = [] # Will hold blocks [S_t, A_t, Y_t] each each time t
    self.y = [] # Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = np.zeros((0, self.L))
    
    # Current network status
    self.current_infected = self.Y[-1,:]
    self.T = 0    
  
  @abstractmethod
  def updateObsHistory(self, a):
    pass
  
  @abstractmethod
  def next_state(self):
    pass
  
  @abstractmethod
  def next_infections(self, a):
    pass
  
  def step(self, a): 
    """
    Move model forward according to action a. 
    :param a: self.L-length array of binary actions at each state 
    """
    self.A = np.vstack((self.A, a))
    self.next_state() 
    self.next_infections(a) 
    self.updateObsHistory(a)
    self.T += 1

  @abstractmethod
  def data_block_at_action(self, data_block, action):
    pass
