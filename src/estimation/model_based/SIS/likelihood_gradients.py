"""
Gradients for (log) likelihood of SIS generative model.
State transitions are parameterized by beta = [beta_0, beta_1].
Infection transitions are parameterized by eta = [eta_0, ..., eta_6].

eta_p0: eta_0, eta_1
eta_p:  eta_2, eta_3, eta_4
eta_q:  eta_5, eta_6
"""
import numpy as np
from scipy.special import expit


# Not implementing this for now since q_l can be solved with vanilla logit regression.
# def exp_logit_q_l(a, eta_q):
#   logit_q_l = eta_q[0] + eta_q[1]*a
#
#
# def q_l_grad(a, exp_logit_q_l_):
#   multiplier = 1 - exp_logit_q_l_ / (1 + exp_logit_q_l_)
#   return np.multiply(multiplier, a)


def logit_gradient(X, y, beta):
  error = y - expit(np.dot(X, beta))
  return np.dot(X.T, error)


def log_p_gradient(eta_p, env):
  X_expit_trt_trt = np.array([1, 1, 1]) * expit(np.sum(eta_p))
  X_expit_trt_notrt = np.array([1, 1, 0]) * expit(np.sum(eta_p[:2]))
  X_expit_notrt_notrt = np.array([1, 0, 0]) * expit(eta_p[0])
  X_expit_notrt_trt = np.array([1, 0, 1]) * expit(np.sum(eta_p[[0,2]]))

  grad_mat = np.column_stack((X_expit_trt_trt, X_expit_trt_notrt, X_expit_notrt_notrt, X_expit_notrt_trt))
  sum_Xy = env.sum_Xy
  trt_pair_vec = env.treat_pair_vec
  return sum_Xy - np.dot(grad_mat, trt_pair_vec)


"""
The gradient with respect to eta_2, eta_3, eta_4 can be found by adding up the number of location-neighbor
pairs with the pattern (trt, trt), (trt, ~trt), (~trt, ~trt), (~trt, trt), respectively.  
"""


def get_treatment_pair_counts(a, env):
  """
  This should be calculated online in SIS.
  """
  trt_trt = 0
  trt_notrt = 0
  notrt_notrt = 0
  notrt_trt = 0

  for l in range(env.L):
    a_l = a[l]
    neighbor_ixs = env.adjacency_list[l]
    num_neighbors = len(neighbor_ixs)
    num_treated_neighbors = np.sum(a[neighbor_ixs])
    if a_l:
      trt_trt += num_treated_neighbors
      trt_notrt += num_neighbors - num_treated_neighbors
    else:
      notrt_trt += num_treated_neighbors
      notrt_notrt += num_neighbors - num_treated_neighbors
  return np.array([trt_trt, trt_notrt, notrt_notrt, notrt_trt])


