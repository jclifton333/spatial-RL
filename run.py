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
  omega = 1
  gamma = 0.9
  featureFunction = polynomialFeatures(3, interaction_only=True)
  # featureFunction = lambda d: d

  if envName == 'SIS':
    g = SIS(L, omega, featureFunction, lattice)
  elif envName == 'Ebola':
    g = Ebola(featureFunction)
  else:
    raise ValueError("Env name not in ['SIS', 'Ebola']")
  # Evaluation limit parameters
  treatment_budget = 5
  evaluation_budget = 5

  # Initialize AR object
  AR = AutoRegressor(LogisticRegression, Ridge)

  means = []
  a_dummy = np.append(np.ones(treatment_budget), np.zeros(g.L - treatment_budget))
  for rep in range(nRep):
    # print('Rep: {}'.format(rep))
    g.reset()
    a = np.random.permutation(a_dummy)
    g.step(a)
    a = np.random.permutation(a_dummy)
    for i in range(T-2):
      # print('i: {}'.format(i))
      g.step(a)
      if method == 'random':
        a = np.random.permutation(a_dummy)
        target = None
      elif method == 'none':
        a = np.zeros(g.L)
        target = None
      else:
        argmax_actions, rollout_feature_list, rollout_Q_function_list, target = rollout(K, gamma, g, evaluation_budget,
                                                                                        treatment_budget, AR,
                                                                                        rollout_feature_times)
        if method == 'QL':
          thetaOpt = GGQ(rollout_feature_list, rollout_Q_function_list, gamma,
                         g, evaluation_budget, treatment_budget, True)
          Q = lambda a: np.dot(
            rollout_Q_features(g.data_block_at_action(g.X_raw[i], a), rollout_Q_function_list, intercept=True),
            thetaOpt)
          _, a, _ = Q_max(Q, evaluation_budget, treatment_budget, g.L)
        elif method == 'rollout':
          a = argmax_actions[-1]
    means.append(np.mean(g.Y[-1,:]))
  return g, AR, means, target


if __name__ == '__main__':
  import time
  n_rep = 10
  for k in range(1, 2):
    t0 = time.time()
    _, _, scores, _ = main(k, 1000, 25, n_rep, 'Ebola', method='none', rollout_feature_times=[0, 1])
    t1 = time.time()
    print('k={}: score={} se={} time={}'.format(k, np.mean(scores), np.std(scores) / np.sqrt(n_rep), t1 - t0))
