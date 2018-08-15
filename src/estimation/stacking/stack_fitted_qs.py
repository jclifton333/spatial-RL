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
        bootstrap_weight_correction_arr[b, t, l] = 1 - exp_
        exp_sum += exp_
      bootstrap_weight_correction_arr[:, t, l] /= exp_sum
  return bootstrap_weight_correction_arr


def stack(q_mb_list, q_mf_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, bootstrap_weight_list,
          intercept=False):
  """

  :param q_mb_list: B-length list of fitted model based q functions
  :param q_mf_list: B-length list of fitted model free q functions
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
  theta = ggq(q_mb_list, q_mf_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, intercept=intercept,
              bootstrap_weight_correction_arr=bootstrap_weight_correction_arr, project=True)
  return theta


def stack_one_step(q_mb_list, q_mf_list, gamma, env, evaluation_budget, treatment_budget, argmaxer,
                   bootstrap_weight_list, intercept=False):
  bootstrap_weight_correction_arr = compute_bootstrap_weight_correction(bootstrap_weight_list)
  y = np.hstack(env.y).astype(float)
  X = np.vstack(env.X)
  X_raw = np.vstack(env.X_raw)
  phi = np.zeros((0, 2))

  # Get targets and features (replicates of correction-weighted y's and X's)
  # (phi refers to q functions as features)
  for b, bootstrap_weight_correction_b in enumerate(bootstrap_weight_correction_arr):
    target_b= np.multiply(y, bootstrap_weight_correction_b)
    phi_b = np.column_stack((q_mb_list[b]()))

  # Get targets (replicates of correction-weighted y's)
  target = np.array([np.multiply(y, bootstrap_correction_b.flatten()) for bootstrap_correction_b in
                     bootstrap_weight_correction_arr])
  target = target.flatten()

  # Get features (replicate sof correction-weighted X's)
  features = np.aray([np.multiply()])

