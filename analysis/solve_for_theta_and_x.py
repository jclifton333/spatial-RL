"""
# Setup: two-stage MDP where Q-function is additive over 2 locations, and the treatment budget is 1 location.
# Stage-1 actions (1, 0), (0, 1) have expected rewards mu.1, mu.2 resp.  Call these actions a1, a2.
# Expected payoff at stage 2 at each location is a linear function (with parameter \theta) of a
# stage-1 covariate and the stage-1 action.
# We want to evaluate the regret of the policy based on the estimator \theta, and in particular how this regret
# varies as \theta approaches boundaries separating the value of different actions.

# MDP
# States at stage 1 are randomly generated iid normal.  Stage 1 expected rewards are known constants mu.1, mu.2
# corresponding
# to the respective actions and independent of state.
# Stages at stage 2 conditional on stage 1 action and states are independent normal with mean vector [x1.i a1.i*x1.i]_
# {i=1,2} %*% theta
# and variance sigma.sq.  Finally, stage 2 rewards are the component of the mean vector corresponding to the location
# that was chosen to
# be treated at stage 2.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from functools import partial
import pdb


def delta_objective(x, theta, eta, delta, mu_1, mu_2, V_opt, sigma_sq):
  """
  Objective function whose solution should give (x, theta, eta) with optimal policy value V_opt and margin delta.

  :param x:
  :param theta:
  :param mu_1:
  :param mu_2:
  :param V_opt:
  :return:
  """
  # Simplify terms
  x1 = x[0, :]
  x2 = x[1, :]
  theta_eta_times_x1 = np.dot(theta + eta, x1)
  theta_eta_times_x2 = np.dot(theta + eta, x2)
  theta_times_x2 = np.dot(theta, x2)

  # p_ij = prob of taking action i at stage 2 given action j was taken at stage 1
  p_11 = norm.cdf(0, loc=theta_eta_times_x1 - theta_times_x2, scale=np.sqrt(2*sigma_sq))
  p_21 = 1 - p_11

  # Value term
  a1_value = mu_1 + theta_eta_times_x1*p_11 * theta_times_x2*p_21
  diff_from_V_opt = (a1_value - V_opt)**2

  # Margin term
  margin = theta_eta_times_x1 - theta_eta_times_x2
  diff_from_delta = (margin - delta)**2

  print('value diff: {} margin diff: {}'.format(diff_from_V_opt, diff_from_delta))
  return diff_from_delta + diff_from_V_opt


def delta_objective_constraint(x, theta, eta, mu_1, mu_2, sigma_sq):
  """
  Solution for (x, theta, eta) must satisfy Q_opt(x, a_1) >= Q_opt(x, a_2).

  :param x:
  :param theta:
  :param eta:
  :param delta:
  :param mu_1:
  :param mu_2:
  :param V_opt:
  :param sigma_sq:
  :return:
  """
  # Simplify terms
  x1 = x[0, :]
  x2 = x[1, :]
  theta_eta_times_x1 = np.dot(theta + eta, x1)
  theta_eta_times_x2 = np.dot(theta + eta, x2)
  theta_times_x1 = np.dot(theta, x1)
  theta_times_x2 = np.dot(theta, x2)

  # p_ij = prob of taking action i at stage 2 given action j was taken at stage 1
  p_11 = norm.cdf(0, loc=theta_eta_times_x1 - theta_times_x2, scale=np.sqrt(2*sigma_sq))
  p_21 = 1 - p_11
  p_12 = norm.cdf(0, loc=theta_eta_times_x2 - theta_times_x1, scale=np.sqrt(2*sigma_sq))
  p_22 = 1 - p_12

  # Q_opt(x, a_i)
  a1_value = mu_1 + theta_eta_times_x1*p_11 * theta_times_x2*p_21
  a2_value = mu_2 + theta_eta_times_x2*p_22 + theta_times_x1*p_12

  return a1_value - a2_value


def delta_objective_constraint_wrapper(parameter, mu_1, mu_2, sigma_sq):
  x = np.array([parameter[[0, 1]], parameter[[2, 3]]])
  theta = parameter[[4, 5]]
  eta = parameter[[6, 7]]
  return delta_objective_constraint(x, theta, eta, mu_1, mu_2, sigma_sq)


def delta_objective_wrapper(parameter, delta, mu_1, mu_2, V_opt, sigma_sq):
  """
  :param parameter: array [x11, x12, x21, x22, theta1, theta2, eta1, eta2].
  :param mu_1:
  :param mu_2:
  :param V_opt:
  :param sigma_sq:
  :return:
  """
  x = np.array([parameter[[0, 1]], parameter[[2, 3]]])
  theta = parameter[[4, 5]]
  eta = parameter[[6, 7]]
  return delta_objective(x, theta, eta, delta, mu_1, mu_2, V_opt, sigma_sq)


def solve_for_theta_and_x(delta, mu_1, mu_2, V_opt, sigma_sq):
  """
  Solve for a gen model parameter (theta (main effect), eta (action effect) and array of stage-1 features (x) such that
    i)  the value at state x is equal to V_opt under the optimal policy, and
    ii) the difference between expected stage-2 rewards under actions 1 and 2 is delta.

  (This allows us to study the effects of varying the margin delta on the value of the _estimated_ optimal policy
  while keeping the value of the true optimal policy fixed.)

  :param mu_1: expected reward for action 1 at stage 1
  :param mu_2: expected reward for action 2 at stage 1
  :param sigma_sq: variance of stage 2 states given stage 1 state and action.
  """
  objective = partial(delta_objective_wrapper, delta=delta, mu_1=mu_1, mu_2=mu_2, V_opt=V_opt, sigma_sq=sigma_sq)
  constraint = partial(delta_objective_constraint_wrapper, mu_1=mu_1, mu_2=mu_2, sigma_sq=sigma_sq)
  ineq_constraint = {'type': 'ineq',
                     'fun': lambda x: constraint(x)}
  res = minimize(objective, np.random.random(size=8), method='SLSQP', constraints=[ineq_constraint])
  param = res.x
  x = np.array([param[[0, 1]], param[[2, 3]]])
  theta = param[[4, 5]]
  eta = param[[6, 7]]
  return {'x': x, 'theta': theta, 'eta': eta, 'res': res}


