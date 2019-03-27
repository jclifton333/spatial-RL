import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
from sklearn.ensemble import RandomForestRegressor
"""
"""


def fit_bandit_pseudo_outcome_models(X1, X2, theta1, theta2, sigma_sq, mc_replicates=100):
  """

  :param X:
  :param theta1:
  :param theta2:
  :return:
  """
  # Draw estimated params at each arm at each replicate
  X1prime_X1 = np.dot(X1.T, X1)
  X2prime_X2 = np.dot(X2.T, X2)
  theta1_hats = np.random.multivariate_normal(theta1, cov=sigma_sq * X1prime_X1, size=mc_replicates)
  theta2_hats = np.random.multivariate_normal(theta2, cov=sigma_sq * X2prime_X2, size=mc_replicates)

  # Generate pseudo-outcomes at each replicate
  X = np.vstack(X1, X2)
  estimated_arm1_means = np.dot(theta1_hats, X)
  estimated_arm2_means = np.dot(theta2_hats, X)

  a1_indicator = estimated_arm1_means.sum(axis=1) >= estimated_arm2_means.sum(axis=1)
  pseudo_outcomes = np.multiply(estimated_arm1_means, a1_indicator) + \
                    np.multiply(estimated_arm2_means, 1 - a1_indicator)
  XA = np.vstack((np.column_stack((X1, np.ones(X1.shape[0]))), np.column_stack((X2, np.ones(X2.shape[0])))))

  # Fit pseudo-outcome models
  models = []
  for rep in range(mc_replicates):
    model = RandomForestRegressor()
    y = pseudo_outcomes[rep]
    model.fit(XA, y)
    models.append(model.predict)

  return models


def evaluate_estimated_bandit_policy(theta1, theta2, x, models):
  regret = 0.0
  x_a1 = np.column_stack((x, np.ones(2)))
  x_a2 = np.column_stack((x, np.zeros(2)))
  r1 = np.sum(np.dot(theta1, x_a1))
  r2 = np.sum(np.dot(theta2, x_a2))
  ropt = np.max((r1, r2))
  for model in models:
    rhat_1 = np.sum(model(x_a1))
    rhat_2 = np.sum(model(x_a2))
    regret += (ropt - rhat_1)*(rhat_1 >= rhat_2) + (ropt - rhat_2)*(rhat_1 < rhat_2)
  regret /= len(models)
  return regret