"""
Gradients for (log) likelihood of SIS generative model.
State transitions are parameterized by beta = [beta_0, beta_1].
Infection transitions are parameterized by eta = [eta_0, ..., eta_6].

eta_p0: eta_0, eta_1
eta_p:  eta_2, eta_3, eta_4
eta_q:  eta_5, eta_6
"""
import pdb
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from functools import partial


def logit_with_label_check(X, y):
  """
  Logistic regression with check for all-0s or all-1s.
  :param X:
  :param y:
  :return:
  """
  y0 = y[0]
  for element in y:
    if element == 1 - y0:
      clf = LogisticRegression()
      clf.fit(X, y)
      return np.append(clf.intercept_, clf.coef_)
  if y0 == 0:
    return np.zeros(X.shape[1] + 1)
  elif y0 == 1:
    return np.zeros(X.shape[1] + 1)


def fit_infection_prob_model(env):
  X = np.vstack(env.X_raw)
  y = np.hstack(env.y)
  infected_ixs = np.where(X[:,2] == 1)
  not_infected_ixs = np.where(X[:,2] == 0)

  A_infected, y_infected = X[infected_ixs, 1], y[infected_ixs]
  A_not_infected, y_not_infected = X[not_infected_ixs, 1], y[not_infected_ixs]

  eta_q = fit_q(A_infected.T, y_infected)
  eta_p0 = fit_p0(A_not_infected.T, y_not_infected)
  eta_p = fit_p(env)
  return np.concatenate((eta_p0, eta_p, eta_q))


def fit_transition_model(env):
  eta = fit_infection_prob_model(env)
  # beta = fit_state_transition_model(env)
  return eta


def fit_q(A_infected, y_infected):
  eta_q = logit_with_label_check(A_infected, y_infected)
  return eta_q


def fit_p0(A_not_infected, y_not_infected):
  eta_q = logit_with_label_check(A_not_infected, y_not_infected)
  return eta_q


def estimate_variance(X, y, fitted_regression_model):
  """
  Get MSE of a simple linear regression.
  """
  y_hat = fitted_regression_model.predict(X.reshape(-1,1))
  n = len(X)
  return np.sum((y - y_hat)**2) / (n - 1)


def fit_state_transition_model(env):
  # ToDO: Compute online (Sherman-Woodbury)
  X = np.vstack(env.X_raw[:-1])
  X_plus = np.vstack(env.X_raw[1:])
  S, S_plus = X[:,0], X_plus[:,0]
  reg = LinearRegression(fit_intercept=False)
  reg.fit(S.reshape(-1,1), S_plus)
  beta_0_hat = reg.coef_[0]
  beta_1_hat = estimate_variance(S, S_plus, reg)
  return beta_0_hat, beta_1_hat


def log_p_gradient(eta_p, env):
  """
  The gradient with respect to eta_2, eta_3, eta_4 can be found by adding up the number of location-neighbor
  pairs with the pattern (trt, trt), (trt, ~trt), (~trt, ~trt), (~trt, trt), respectively.  We keep track
  of the relevant information in env for efficiency and combine it here to get the gradient.

  ToDo: Limit to uninfected sites!
  """
  X_expit_trt_trt = np.array([1, 1, 1]) * expit(np.sum(eta_p))
  X_expit_trt_notrt = np.array([1, 1, 0]) * expit(np.sum(eta_p[:2]))
  X_expit_notrt_notrt = np.array([1, 0, 0]) * expit(eta_p[0])
  X_expit_notrt_trt = np.array([1, 0, 1]) * expit(np.sum(eta_p[[0,2]]))

  grad_mat = np.column_stack((X_expit_trt_trt, X_expit_trt_notrt, X_expit_notrt_notrt, X_expit_notrt_trt))
  sum_Xy = env.sum_Xy
  trt_pair_vec = env.treat_pair_vec
  return sum_Xy - np.dot(grad_mat, trt_pair_vec)


def fit_p(env, warm_start = None, tol=1e-4, max_iter=100):
  grad_func = partial(log_p_gradient, env=env)
  if warm_start is None:
    eta_p = np.zeros(3)
  else:
    eta_p = warm_start
  step_size = 1/len(env.X_raw)
  iter = 0
  while iter < max_iter:
    new_eta_p = eta_p + step_size*grad_func(eta_p)
    diff = np.linalg.norm(new_eta_p - eta_p) / (np.linalg.norm(eta_p) + 1)
    if diff < tol:
      break
    eta_p = new_eta_p
    iter += 1
  return eta_p





