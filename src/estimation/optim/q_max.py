import numpy as np
from src.estimation.q_functions import q
from src.estimation.optim.sweep import argmaxer_sweep


def q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model, q_max=argmaxer_sweep.sweep,
                     network_features=False):
  """
  Take q_max for all data blocks in env history (need this for Q-learning/rollout).
  """
  best_q_arr = []
  argmax_data_blocks = []
  argmax_actions = []
  for t in range(env.T):
    q_fn_t = lambda a: q(a, env.X_raw[t], env, predictive_model, network_features=network_features)
    q_max_t, q_argmax_t, q_vals = q_max(q_fn_t, evaluation_budget, treatment_budget, env.L)
    best_q_arr.append(q_max_t)
    best_data_block = env.data_block_at_action(env.X_raw[t], q_argmax_t)
    argmax_data_blocks.append(best_data_block)
    argmax_actions.append(q_argmax_t)
  return np.array(best_q_arr), argmax_data_blocks, argmax_actions, q_vals
