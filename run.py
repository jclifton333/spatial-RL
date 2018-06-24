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
from src.environments.environment_factory import environment_factory

from src.policies.Policy import policy_factory

from src.estimation.AutoRegressor import AutoRegressor
from src.estimation.Fitted_Q import rollout, rollout_Q_features
from src.estimation.Q_functions import Q_max
from src.estimation.Greedy_GQ import GGQ

from src.utils.features import polynomialFeatures

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from src.utils.misc import RidgeProb


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


def main(lookahead_depth, T, nRep, env_name, policy_name, **kwargs):
  """
  :param lookahead_depth:
  :param env_name: 'SIS' or 'Ebola'
  :param T: duration of simulation rep
  :param nRep: number of replicates
  :param policy_name: string in ['random', 'no_action', 'true_probs', 'rollout', 'network rollout'].
  :param kwargs: environment-specific keyword arguments
  """
  # Initialize generative model
  gamma = 0.7
  # feature_function = polynomialFeatures(3, interaction_only=True)

  def feature_function(x):
    return x

  env = environment_factory(env_name, feature_function, **kwargs)

  # Evaluation limit parameters
  treatment_budget = np.int(np.floor((3/16) * kwargs['L']))
  evaluation_budget = 20

  policy = policy_factory(policy_name)
  true_probs_policy = policy_factory('true_probs')
  random_policy = policy_factory('random')
  policy_arguments = {'classifier': RidgeProb, 'regressor':RandomForestRegressor, 'env':env,
                      'evaluation_budget':evaluation_budget, 'gamma':gamma, 'rollout_depth':lookahead_depth,
                      'treatment_budget':treatment_budget}
  score_list = []
  for rep in range(nRep):
    env.reset()
    for t in range(T):
      t0 = time.time()
      print('rep: {} t: {}'.format(rep, t))
      a = policy(**policy_arguments)
      env.step(a)
      print('a random score: {} a est score: {} a true score: {}'.format(
        np.mean(env.next_infected_probabilities(random_policy(**policy_arguments))),
        np.mean(env.next_infected_probabilities(a)),
        np.mean(env.next_infected_probabilities(true_probs_policy(**policy_arguments)))))
      t1 = time.time()
      print('Time: {}'.format(t1 - t0))
    score_list.append(np.mean(env.Y))
    print('Episode score: {}'.format(np.mean(env.Y)))
  return score_list


if __name__ == '__main__':
  import time
  n_rep = 1
  SIS_kwargs = {'L': 16, 'omega': 0, 'generate_network': lattice}
  for k in range(0, 1):
    t0 = time.time()
    scores = main(k, 5, n_rep, 'SIS', 'rollout', **SIS_kwargs)
    t1 = time.time()
    print('k={}: score={} se={} time={}'.format(k, np.mean(scores), np.std(scores) / np.sqrt(n_rep), t1 - t0))
