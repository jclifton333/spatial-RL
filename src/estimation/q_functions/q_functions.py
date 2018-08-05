# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import pdb
import numpy as np 
"""
Parameter descriptions 

evaluation budget: number of treatment combinations to evaluate
treatment_budget: size of treated state subset
"""


def q(a, data_block_ix, env, predictive_model, raw=False):
  data_block = env.data_block_at_action(data_block_ix, a, raw=raw)
  q_hat = predictive_model(data_block)
  return q_hat

    

  
  



