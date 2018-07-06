"""
Reference policies for comparison:
  - none
  - random
  - true probabilities
"""
import numpy as np
import pdb
from src.utils.misc import random_argsort


def treat_all(**kwargs):
  L = kwargs['env'].L
  return np.ones(L)


def no_action(**kwargs):
  L = kwargs['env'].L
  return np.zeros(L)


def true_probs(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer']
  a = argmaxer(env.next_infected_probabilities, evaluation_budget, treatment_budget, env)
  return a


def true_probs_myopic(**kwargs):
  env, treatment_budget, divide_evenly = kwargs['env'], kwargs['treatment_budget'], kwargs['divide_evenly']
  a = np.zeros(env.L)
  probs = env.next_infected_probabilities(a)
  if divide_evenly:
    infected_ixs = np.where(env.current_infected == 1)[0]
    not_infected_ixs = np.where(env.current_infected == 0)[0]
    allocate_to_infected = np.int(np.floor(treatment_budget / 2)) + \
                           (np.random.random() < 0.5)*(treatment_budget % 2 == 0)
    allocate_to_not_infected = treatment_budget - allocate_to_infected
    treat_infected_ixs = random_argsort(-probs[infected_ixs], allocate_to_infected)
    treat_not_infected_ixs = random_argsort(-probs[not_infected_ixs], allocate_to_not_infected)
    a[infected_ixs][treat_infected_ixs] = 1
    a[not_infected_ixs][treat_not_infected_ixs] = 1
  else:
    treat_ixs = random_argsort(-probs, treatment_budget)
    a[treat_ixs] = 1
  return a


def random(**kwargs):
  treatment_budget, divide_evenly = kwargs['treatment_budget'], kwargs['divide_evenly']

  if divide_evenly: # Split between infected and not-infected states
    current_infected = kwargs['env'].current_infected
    infected_ixs = np.where(current_infected == 1)
    not_infected_ixs = np.where(current_infected == 0)
    try:
      if np.random.random() < 0.5:
        infected_treatment_budget = np.min([np.int(np.floor(treatment_budget / 2)),
                                           len(infected_ixs[0])])
      else:
        infected_treatment_budget = np.min([np.int(np.ceil(treatment_budget / 2)),
                                           len(not_infected_ixs[0])])
      not_infected_treatment_budget = np.min([treatment_budget - infected_treatment_budget,
                                             len(not_infected_ixs[0])])
    except:
      pdb.set_trace()
    infected_trts = np.random.choice(infected_ixs[0], infected_treatment_budget)
    not_infected_trts = np.random.choice(not_infected_ixs[0], not_infected_treatment_budget)
    a = np.zeros_like(current_infected)
    a[infected_trts] = 1
    a[not_infected_trts] = 1
  else:
    L = kwargs['env'].L
    assert treatment_budget < L
    dummy_act = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
    a = np.random.permutation(dummy_act)
  return a