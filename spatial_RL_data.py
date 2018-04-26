# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:14:03 2018

@author: Jesse
"""

import numpy as np

class spatial_RL_data(object):
  '''
  Class for storing and manipulating data for spatial RL.
  Should facilitate location-level (e.g. individual location probabilites)
  and network-level (e.g. expected number of infections) prediction problems.
  '''
  
  def __init__(self, L, nCov):
    '''
    :param L: number of locations 
    :param nCov: covariate dimension
    '''
    self.S = np.zeros((0, L, nCov)) 
    self.A = np.zeros((0, L))
    self.Y = np.zeros((0, L))
    
  def update(self, a, y, s):
    '''
    Add observations to data arrays.
    :param a: L-length binary array of actions
    :param y: L-length binary array of infection indicators
    :param s: LxnCov-array of location covariates
    '''
    self.A = np.vstack((self.A, a))
    self.Y = np.vstack((self.Y, y))
    self.S = np.concatenate((self.S, s[None,:]), axis=0)