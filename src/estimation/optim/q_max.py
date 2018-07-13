import pdb
import numpy as np
from src.estimation.q_functions.q_functions import q


def q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model, argmaxer,
                     ixs, network_features=False, return_blocks_at_argmax=False):
  """
  Take q_max for all data blocks in env history (need this for Q-learning/rollout).
  """
  q_max_list = []
  argmax_list = []
  if return_blocks_at_argmax:
    block_at_argmax_list = []
  else:
    block_at_argmax_list = None
  for t in range(env.T):
    if ixs is None:
      q_fn = lambda a: q(a, t, env, predictive_model, network_features=network_features)
      argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env)
      if return_blocks_at_argmax:
        block_at_argmax = env.data_block_at_action(t, argmax, ixs=None)
        block_at_argmax_list.append(block_at_argmax)
    else:
      q_fn = lambda a: q(a, t, env, predictive_model, ixs[t], network_features=network_features)
      argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env, ixs=ixs[t])
      if return_blocks_at_argmax:
        block_at_argmax = env.data_block_at_action(t, argmax, ixs=ixs[t])
        block_at_argmax_list.append(block_at_argmax)
    argmax_list.append(argmax)
    q_max_list.append(q_fn(argmax))
  return np.array(q_max_list), argmax_list, block_at_argmax_list
