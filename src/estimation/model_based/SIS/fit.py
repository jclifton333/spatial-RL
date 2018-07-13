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
from functools import partial
from .p_objective import negative_log_likelihood
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.optimize import minimize


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
    return np.ones(X.shape[1] + 1)


def fit_infection_prob_model(env, ixs):
  """

  :param env:
  :param ixs: List of lists of indexes for train subset (used for stacking method).
  :return:
  """
  if ixs is None:
    X = np.vstack(env.X_raw)
    y = np.hstack(env.y)
    counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected = \
      env.counts_for_likelihood_next_infected, env.counts_for_likelihood_next_not_infected
  else:
    X = np.vstack([env.X_raw[t][ixs[t], :] for t in range(len(env.X_raw))])
    y = np.hstack([env.y[t][ixs[t]] for t in range(len(env.y))])
    counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected = \
      env.get_likelihood_information_for_cv_split(ixs)

  infected_ixs = np.where(X[:, 2] == 1)
  A_infected, y_infected = X[infected_ixs, 1], y[infected_ixs]

  eta_q = fit_q(A_infected.T, y_infected)
  eta_p = fit_p(env, counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected)
  return np.concatenate((eta_p, eta_q))


def fit_transition_model(env, ixs=None):
  eta = fit_infection_prob_model(env, ixs)
  # beta = fit_state_transition_model(env)
  return eta


def fit_q(A_infected, y_infected):
  eta_q = logit_with_label_check(A_infected, 1 - y_infected)
  return eta_q


def fit_p(env, counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected):
  """
  ToDo: Document these parameters.
  :param counts_for_likelihood_next_infected:
  :param counts_for_likelihood_next_not_infected:
  :return:
  """
  objective = partial(negative_log_likelihood, counts_for_likelihood_next_infected=counts_for_likelihood_next_infected,
                      counts_for_likelihood_next_not_infected=counts_for_likelihood_next_not_infected)
  res = minimize(objective, x0=env.eta[:5], method='L-BFGS-B')
  eta_p = res.x
  return eta_p


def estimate_variance(X, y, fitted_regression_model):
  """
  Get MSE of a simple linear regression.
  """
  y_hat = fitted_regression_model.predict(X.reshape(-1,1))
  n = len(X)
  return np.sum((y - y_hat)**2) / (n - 1)


def fit_state_transition_model(env):
  # This is not actually used in estimating model assuming omega=0.
  # ToDO: Compute online (Sherman-Woodbury)
  X = np.vstack(env.X_raw[:-1])
  X_plus = np.vstack(env.X_raw[1:])
  S, S_plus = X[:,0], X_plus[:,0]
  reg = LinearRegression(fit_intercept=False)
  reg.fit(S.reshape(-1,1), S_plus)
  beta_0_hat = reg.coef_[0]
  beta_1_hat = estimate_variance(S, S_plus, reg)
  return beta_0_hat, beta_1_hat



