# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:23:02 2018

@author: Jesse
"""

import numpy as np
from SIS_model import SIS
from generate_network import lattice
from scipy.optimize import minimize
import pdb

def prop_infected_error(sigma, omega, L, T, prop_infected):
  '''
  Measure how close an SIS model with given settings comes to 
  prop_infected.
  '''
  adjacency_matrix = lattice(L)  
  g = SIS(adjacency_matrix, omega, sigma)
  prop_infected_list = []
  for rep in range(20):
    for t in range(T):
      g.step(np.zeros(g.nS))
    prop_infected_list.append(np.mean(g.Y[-1,:]))
  return np.abs(prop_infected - np.mean(prop_infected_list))
    

def tune_SIS_hyperparameters(omega, L, T, prop_infected):
  '''
  Find the hyperparameter setting closest to prop_infected after
  T steps.  
  '''
  objective = lambda sigma: prop_infected_error(sigma, omega, L, T, prop_infected)
  
  #Create evaluation grid
  #There's gotta be a better way to do this
  l = np.logspace(-1, 1, 3)
  m = np.meshgrid(l, l, l, l, l, l, l)
  grid = np.vstack((i.flatten() for i in m)).T
  for i in range(grid.shape[0]):
    val = objective(grid[i,:])
    print(val)
    if val < 0.01:
      return grid[i,:]
  return None

s = tune_SIS_hyperparameters(0.5, 25, 50, 0.5)

