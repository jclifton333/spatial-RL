import numpy as np
from .greedy_gq import ggq


def compute_bootstrap_weight_correction(bootstrap_weights_list):
  """
  Compute corrections associated with bootstrap weights, in order to perform multiplier bootstrap prediction error
  estimation.

  :param bootstrap_weights_list: B-length list of (T x L) arrays of bootstrap weights.
  :return:
  """
  T, L = bootstrap_weights_list[0].shape
  B = len(bootstrap_weights_list)
  bootstrap_weight_correction_arr = np.zeros((B, T, L))
  for t in range(T):
    for l in range(L):
      exp_sum = 0
      for b in range(B):
        exp_ = np.exp(-bootstrap_weights_list[b][t, l])
        bootstrap_weight_correction_arr[b, t, l] = exp_
        exp_sum += exp_
      bootstrap_weight_correction_arr[:, t, l] /= exp_sum
  return bootstrap_weight_correction_arr


def stack(q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, bootstrap_weight_list,
          intercept=False):
  """

  :param q1_list: B-length list of fitted q functions
  :param q2_list: B-length list of fitted q functions
  :param gamma:
  :param env:
  :param evaluation_budget:
  :param treatment_budget:
  :param argmaxer:
  :param bootstrap_weight_list: B-length list of TxL arrays of bootstrap weights that were used to fit the q functions.
  :param intercept:
  :return:
  """
  bootstrap_weight_correction_arr = compute_bootstrap_weight_correction(bootstrap_weight_list)
  theta = ggq(q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, intercept=True,
              bootstrap_weight_correction_arr=bootstrap_weight_correction_arr, project=True)
  return theta

