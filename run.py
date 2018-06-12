# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import sys
# sys.path.append('src/environments')
# sys.path.append('src/estimation')
# sys.path.append('src/utils')

import numpy as np
import pdb

from src.environments.generate_network import lattice
from src.environments.Ebola import Ebola
from src.environments.SIS import SIS

from src.estimation.AutoRegressor import AutoRegressor
from src.estimation.Fitted_Q import rollout, rollout_Q_features
from src.estimation.Q_functions import Q_max
from src.estimation.Greedy_GQ import GGQ

from src.utils.features import polynomialFeatures

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression


# def divide_random_between_infection_status(treatment_budget, current_infected):
#   """
#   Return an action with _treatment_budget_ treatments divided evenly between
#   infected and not-infected states.
#
#   :param treatment_budget:
#   :param current_infected:
#   :return:
#   """
#   infected_ixs = np.where(current_infected == 1)
#   not_infected_ixs = np.where(current_infected == 0)
#   # pdb.set_trace()
#   try:
#     if np.random.random() < 0.5:
#       infected_treatment_budget = np.min([np.int(np.floor(treatment_budget / 2)),
#                                          len(infected_ixs[0])])
#     else:
#       infected_treatment_budget = np.min([np.int(np.ceil(treatment_budget / 2)),
#                                          len(not_infected_ixs[0])])
#     not_infected_treatment_budget = np.min([treatment_budget - infected_treatment_budget,
#                                            len(not_infected_ixs[0])])
#   except:
#     pdb.set_trace()
#   infected_trts = np.random.choice(infected_ixs[0], infected_treatment_budget)
#   not_infected_trts = np.random.choice(not_infected_ixs[0], not_infected_treatment_budget)
#   a = np.zeros_like(current_infected)
#   a[infected_trts] = 1
#   a[not_infected_trts] = 1
#   return a


def main(K, L, T, nRep, envName, method='QL', rollout_feature_times=[1]):
  """
  :param K: lookahead depth
  :param L: number of locations in network
  :param envName: 'SIS' or 'Ebola'
  :param T: duration of simulation rep
  :param nRep: number of replicates
  :param method: string in ['QL', 'rollout', 'random', 'none']
  """
  # Initialize generative model
  omega = 0
  gamma = 0.7
  # featureFunction = polynomialFeatures(3, interaction_only=True)
  featureFunction = lambda d: d

  if envName == 'SIS':
    g = SIS(L, omega, featureFunction, lattice)
  elif envName == 'Ebola':
    g = Ebola(featureFunction)
  else:
    raise ValueError("Env name not in ['SIS', 'Ebola']")
  # Evaluation limit parameters
  treatment_budget = np.int(np.floor((3/16) * L))
  evaluation_budget = 20

  # Initialize AR object
  AR = AutoRegressor(RandomForestClassifier, RandomForestRegressor)

  means = []
  a_dummy = np.append(np.ones(treatment_budget), np.zeros(g.L - treatment_budget))
  for rep in range(nRep):
    print('Rep: {}'.format(rep))
    g.reset()
    a = np.random.permutation(a_dummy)
    g.step(a)
    a = np.random.permutation(a_dummy)
    mean_disagreement = 0
    for i in range(T-2):
      # print('i: {}'.format(i))
      g.step(a)
      if method == 'random':
        a = np.random.permutation(a_dummy)
        # a = divide_random_between_infection_status(treatment_budget, g.current_infected)
        target = None
      elif method == 'none':
        a = np.zeros(g.L)
        target = None
      elif method == 'true-probs':
        _, a, _ = Q_max(g.next_infected_probabilities, evaluation_budget, treatment_budget, g.L)
        target = None
      else:
        argmax_actions, rollout_feature_list, rollout_Q_function_list, target, r2 = rollout(K, gamma, g, evaluation_budget,
                                                                                        treatment_budget, AR,
                                                                                        rollout_feature_times)
        print(r2)
        if method == 'QL':
          thetaOpt = GGQ(rollout_feature_list, rollout_Q_function_list, gamma,
                         g, evaluation_budget, treatment_budget, True)
          Q = lambda a: np.dot(
            rollout_Q_features(g.data_block_at_action(g.X_raw[i], a), rollout_Q_function_list, intercept=True),
            thetaOpt)
          _, a, _ = Q_max(Q, evaluation_budget, treatment_budget, g.L)
        elif method == 'rollout':
          a = argmax_actions[-1]
          # Compare with true-probs action
          _, a_true, _ = Q_max(g.next_infected_probabilities, evaluation_budget, treatment_budget, g.L)
          print('a random score: {} a est score: {} a true score: {}'.format(np.mean(g.next_infected_probabilities(np.random.permutation(a_dummy))),
                                                                             np.mean(g.next_infected_probabilities(a)),
                                                                              np.mean(g.next_infected_probabilities(a_true))))
    print('mean disagrement: {}'.format(mean_disagreement))
    means.append(np.mean(g.Y))
  return g, AR, means, target


if __name__ == '__main__':
  import time
  n_rep = 5
  for k in range(0, 1):
    t0 = time.time()
    _, _, scores, _ = main(k, 16, 100, n_rep, 'SIS', method='rollout', rollout_feature_times=[0, 1])
    t1 = time.time()
    print('k={}: score={} se={} time={}'.format(k, np.mean(scores), np.std(scores) / np.sqrt(n_rep), t1 - t0))
