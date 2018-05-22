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

'''
Tune SIS model acc to new manuscript,
pdf pg. 15
'''

def tune_SIS():
  #Fixed S, Y, and A values
  S_any = S_pos = np.ones(4)
  Y_p10 = np.zeros(4)
  Y_p1  = np.array([0, 1, 1, 1])
  Y_q1  = np.ones(4)
  A_p10  = A_p1_1 = np.zeros(4)
  A_p1_1 = np.array([1, 0, 0, 0])
  A_p1_2 = 1 - A_p1_1
  A_q1_1 = np.zeros(4)
  A_q1_2 = np.ones(4)
  
  '''
  Equations
  (EITHER I'M DUMB OR EQUATIONS IN PAPER DON'T MAKE SENSE)
  \eta_0 = logit(0.01)
  \eta_2 = logit[1 - ((1 - 0.5) / 0.99)**(1/3)]
  \eta_2 + \eta_3 = logit[1 - ((1 - 0.5*0.75) / (1 - 0.01*expit(\eta_1)))**(1/3)]
  \eta_2 + \eta_4 = logit[1 - ((1 - 0.5*25) / 0.99)**(1/3)]
  \eta_5 = 0.25
  \eta_6 = 0.25 * 0.5
  '''
  return















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

