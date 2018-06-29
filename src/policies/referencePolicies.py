"""
Reference policies for comparison:
  - none
  - random
  - true probabilities
"""
import numpy as np
from src.estimation.Q_functions import Q_max


def treat_all(**kwargs):
  L = kwargs['env'].L
  return np.ones(L)


def no_action(**kwargs):
  L = kwargs['env'].L
  return np.zeros(L)


def random(**kwargs):
  L, treatment_budget = kwargs['env'].L, kwargs['treatment_budget']
  assert treatment_budget < L
  dummy_act = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
  return np.random.permutation(dummy_act)


def true_probs(**kwargs):
  env, treatment_budget, evaluation_budget = kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget']
  _, a, _ = Q_max(env.next_infected_probabilities, evaluation_budget, treatment_budget, env.L)
  return a


def true_probs_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  a = np.zeros(env.L)
  probs = env.next_infected_probabilities(a)
  treat_ixs = np.argsort(-probs)[:treatment_budget]
  a[treat_ixs] = 1
  return a

