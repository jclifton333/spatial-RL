import numpy as np
import pdb
from scipy.optimize import minimize
from numba import njit, jit
from functools import partial


@njit
def negative_log_likelihood(eta0, exp_eta1, exp_eta2, eta3, eta4, A, Y, y_next, distance_matrix, product_matrix, T, L):
  log_lik = 0
  for t in range(T):
    a = A[t, :]
    y = Y[t, :]
    y_next_ = y_next[t, :]

    for l in range(L):
      if not y[l]:  # Only model uninfected locations; infected locations never recover

        # Log likelihood at location l at time t
        prod = 1.0
        a_l = a[l]
        for l_prime in range(L):
          d_l_lprime = distance_matrix[l, l_prime]
          s_l_lprime = product_matrix[l, l_prime]
          a_l_prime = a[l_prime]
          logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.power(s_l_lprime, exp_eta2) + \
            eta3 * a_l + eta4 * a_l_prime
          transmission_prob = 1.0 - 1.0 / (1.0 + np.exp(logit_transmission_prob))
          prod *= (1 - transmission_prob)
        if y_next_[l]:
          log_lik += np.log(1 - prod)
        else:
          log_lik += np.log(prod)

  return -log_lik


# ToDo: This turns out to be unnecessary...
def negative_log_likelihood_wrapper(eta, A, Y, y_next, distance_matrix, product_matrix, T, L):
  eta0 = eta[0]
  exp_eta1 = np.exp(eta[1])
  exp_eta2 = np.exp(eta[2])
  eta3 = eta[3]
  eta4 = eta[4]
  return negative_log_likelihood(eta0, exp_eta1, exp_eta2, eta3, eta4, A, Y, y_next, distance_matrix, product_matrix,
                                 T, L)


def fit_ebola_transition_model(env):
  objective = partial(negative_log_likelihood_wrapper, A=env.A, Y=env.Y, y_next=np.array(env.y),
                      distance_matrix=env.DISTANCE_MATRIX, product_matrix=env.PRODUCT_MATRIX, T=env.T, L=env.L)
  res = minimize(objective, x0=env.ETA, method='L-BFGS-B')
  eta_hat = res.x
  return eta_hat


# @njit
# def log_lik_at_location(eta0, exp_eta1, exp_eta2, eta3, eta4, l, a, y_next, distance_matrix, product_matrix, L):
#   prod = 1.0
#   a_l = a[l]
#   for l_prime in range(L):
#     d_l_lprime = distance_matrix[l, l_prime]
#     s_l_lprime = product_matrix[l, l_prime]
#     a_l_prime = a[l_prime]
#     logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.power(s_l_lprime, exp_eta2) + \
#       eta3 * a_l + eta4 * a_l_prime
#     transmission_prob = 1.0 - 1.0 / (1.0 + np.exp(logit_transmission_prob))
#     prod *= (1 - transmission_prob)
#   if y_next[l]:
#     return np.log(1 - prod)
#   else:
#     return np.log(prod)
