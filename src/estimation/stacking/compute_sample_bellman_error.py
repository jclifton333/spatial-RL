import numpy as np
from src.estimation.q_functions.q_functions import q_max_all_states


def compute_temporal_differences(q_fn, gamma, env, evaluation_budget, treatment_budget, argmaxer,
                                 bootstrap_correction_weights=None, q_of_X=None, ixs=None):
  """
  TD = Y + \gamma * q_max_of_Xp1 - q_of_X.

  :param q_fn:
  :param gamma:
  :param env:
  :param evaluation_budget:
  :param treatment_budget:
  :param argmaxer:
  :param bootstrap_correction_weights: (T x L) array of weights to correct for bootstrapping
  :param q_of_X: q_fn evaluated at env data blocks; calculated in function if not be provided.
                 Should be provided for GGQ.
  :param ixs:
  :return:
  """
  assert env.T > 1

  # Evaluate Q
  if q_of_X is None:
    q_of_X = np.array([])
    for data_block in env.X[:-1]:
      q_of_X = np.append(q_of_X, q_fn(data_block))

  # blocks_at_argmax_list is list of [S, A, Y] blocks where A = argmax_a q_fn( S, a, Y)
  q_max_of_Xp1, argmax_list, blocks_at_argmax_list = q_max_all_states(env, evaluation_budget, treatment_budget,
                                                                      q_fn, argmaxer, ixs,
                                                                      return_blocks_at_argmax=True)
  q_max_of_Xp1 = q_max_of_Xp1[1:, ]

  # Compute TD * semi-gradient and q features at argmax (X_hat)
  if ixs is not None:
    y = [env.y[t][ixs[t]] for t in range(len(env.y[:-1]))]
  else:
    y = env.y[:-1]

  TD = np.hstack(y).astype(float) + gamma * q_max_of_Xp1.flatten() - q_of_X
  TD = TD.reshape(TD.shape[0], 1)
  if bootstrap_correction_weights is not None:
    TD = np.multiply(TD, bootstrap_correction_weights.T.flatten())
  TD_times_q_of_X = np.multiply(TD.T, q_of_X)
  return TD, TD_times_q_of_X, blocks_at_argmax_list[1:]


def compute_sample_squared_bellman_error(q_fn, gamma, env, evaluation_budget, treatment_budget, argmaxer, ixs=None):
  TD, _, _ = compute_temporal_differences(q_fn, gamma, env, evaluation_budget, treatment_budget, argmaxer, ixs=ixs)
  return np.mean(TD**2)
