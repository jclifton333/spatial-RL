# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:06:36 2018

@author: Jesse
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def polynomialFeatures(num_raw_features, interaction_only):
  poly = PolynomialFeatures(interaction_only=interaction_only)
  dummy = np.zeros((1, num_raw_features))
  poly.fit_transform(dummy)
  return poly.transform  
  

  
  
  