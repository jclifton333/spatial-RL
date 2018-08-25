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


def q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model, argmaxer,
                     return_blocks_at_argmax=False, raw=False):
  """
  Take q_max for all data blocks in env history (need this for Q-learning/fqi).
  """
  q_max_list = []
  argmax_list = []
  if return_blocks_at_argmax:
    block_at_argmax_list = []
  else:
    block_at_argmax_list = None
  for t in range(env.T):
    q_fn = lambda a: q(a, t, env, predictive_model, raw=raw)
    argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env)
    if return_blocks_at_argmax:
      block_at_argmax = env.data_block_at_action(t, argmax, raw=raw)
      block_at_argmax_list.append(block_at_argmax)
    argmax_list.append(argmax)
    q_max_list.append(q_fn(argmax))
  return np.array(q_max_list), argmax_list, block_at_argmax_list