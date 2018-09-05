"""
From draft:

1. Tune \alpha such that {\alpha * \eta_0 MLE, log(\alpha) + \eta_1 MLE, \eta_2 MLE, 0, 0} gives 70% infections after
   25 time points under no treatment
2. Tune \beta such that {\alpha \eta_0 MLE, log(\alpha) + \eta_1 MLE, \eta_2 MLE, \beta, \beta} such that the increase
   in infections after 25 time points under the all-treatment policy is 0.05 times that under the no-treatment policy
"""
import sys
import os
this_dirname = os.path.dirname(os.path.abspath(__file__))
src_dirname = os.path.join(this_dirname, '..', '..', '..')
sys.path.append(src_dirname)
from src.environments.Ebola import Ebola
import numpy as np
from scipy.optimize import minimize


def alpha_loss(env, Y_25):
  observed_infections = np.mean(np.sum(Y_25, axis=1))
  target_infections = 0.7 * env.L
  return np.abs(target_infections - observed_infections)


def beta_loss(Y_25_treat_all, Y_25_treat_none, Y_0_mean):
  observed_infections_treat_all = np.mean(np.sum(Y_25_treat_all, axis=1))
  observed_infections_treat_one = np.mean(np.sum(Y_25_treat_none, axis=1))
  return np.abs((observed_infections_treat_all - Y_0_mean) / (observed_infections_treat_one - Y_0_mean) - 0.05)


def loss(env, Y_25, Y_25_treat_all, Y_25_treat_none, Y_0_mean):
  return alpha_loss(env, Y_25) + beta_loss(Y_25_treat_all, Y_25_treat_none, Y_0_mean)


def alpha_objective(log_alpha):
  alpha = np.exp(log_alpha)
  eta_alpha = np.array([Ebola.ETA_0 * alpha, np.log(alpha) + Ebola.ETA_1, Ebola.ETA_2, 0.0, 0.0])
  env = Ebola(eta=eta_alpha)

  Y_25 = np.zeros((0, env.L))
  for i in range(10):
    for t in range(23):
      env.step(np.zeros(env.L))
    Y_25 = np.vstack((Y_25, env.current_infected))

  return alpha_loss(env, Y_25)


def beta_objective(alpha, log_beta):
  beta = -np.exp(log_beta)
  eta_beta = np.array([Ebola.ETA_0 * alpha, np.log(alpha) + Ebola.ETA_1, Ebola.ETA_2, beta, beta])
  env = Ebola(eta=eta_beta)

  Y_0_mean = np.sum(env.Y[0, :])

  # treat-none policy
  Y_25_treat_none = np.zeros((0, env.L))
  for i in range(100):
    for t in range(23):
      env.step(np.zeros(env.L))
    Y_25_treat_none = np.vstack((Y_25_treat_none, env.current_infected))

  # treat-all policy
  Y_25_treat_all = np.zeros((0, env.L))
  for i in range(100):
    for t in range(23):
      env.step(np.ones(env.L))
    Y_25_treat_all = np.vstack((Y_25_treat_all, env.current_infected))

  return beta_loss(Y_25_treat_all, Y_25_treat_none, Y_0_mean)


def tune():
  log_alpha = minimize(alpha_objective, x0=[0.0], method='L-BFGS-B').x
  alpha = np.exp(log_alpha)
  print(alpha)
  log_beta = minimize(lambda b: beta_objective(alpha, b), x0=[0.0], method='L-BFGS-B').x
  beta = np.exp(log_beta)
  print(alpha, beta)
  return alpha, beta


if __name__ == '__main__':
  # alpha_list = np.array([1.0, 1.5, 2.0])
  # for alpha in alpha_list:
  #   loss = alpha_objective(np.log(alpha))
  #   print('alpha {} loss {}'.format(alpha, loss))
  tune()

