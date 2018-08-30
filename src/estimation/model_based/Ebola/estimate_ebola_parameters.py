import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from functools import partial


def log_lik_at_location(eta0, exp_eta1, exp_eta2, eta3, eta4, l, a, y_next, env):
  prod = 1.0
  a_l = a[l]
  for l_prime in range(env.L):
    d_l_lprime = env.DISTANCE_MATRIX[l, l_prime]
    s_l_lprime = env.PRODUCT_MATRIX[l, l_prime]
    a_l_prime = a[l_prime]
    logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.exp(s_l_lprime, exp_eta2) + \
      eta3 * a_l + eta4 * a_l_prime
    transmission_prob = expit(logit_transmission_prob)
    prod *= (1 - transmission_prob)
  if y_next:
    return np.log(1 - prod)
  else:
    return np.log(prod)


def negative_log_likelihood(eta, env):
  eta0 = eta[0]
  exp_eta1 = np.exp(eta[1])
  exp_eta2 = np.exp(eta[2])
  eta3 = eta[3]
  eta4 = eta[4]

  log_lik = 0
  for t in range(env.T):
    a = env.A[t]
    y = env.Y[t]
    y_next = env.y[t]
    for l in range(env.L):
      if not y[l]:  # Only model uninfected locations; infected locations never recover
        log_lik += log_lik_at_location(eta0, exp_eta1, exp_eta2, eta3, eta4, l, a, y_next, env)
  return -log_lik


def fit_ebola_transition_model(env):
  objective = partial(negative_log_likelihood, env=env)
  res = minimize(objective, x0=env.ETA, method='L-BFGS-B')
  eta_hat = res.x
  return eta_hat
