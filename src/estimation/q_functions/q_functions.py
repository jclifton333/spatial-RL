# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import numpy as np 
"""
Parameter descriptions 

evaluation budget: number of treatment combinations to evaluate
treatment_budget: size of treated state subset
"""


def q(a, raw_data_block, env, predictive_model, network_features=False):
  if network_features:
    data_block = env.network_features_at_action(raw_data_block, a)
  else:
    data_block = env.data_block_at_action(raw_data_block, a)
  q_hat = predictive_model(data_block)
  return q_hat

    

  
  



