import numpy as np
import copy
import pdb

# Hack bc python imports are stupid
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', 'src')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory
from src.utils.misc import KerasLogit, KerasRegressor


def run_sims_for_bootstrap_dbns(rollout_depth, T, n_rep, argmaxer_name, **kwargs):
  """
  :param rollout_depth:
  :param T: duration of simulation rep
  :param n_rep: number of replicates
  :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
  :param kwargs: environment-specific keyword arguments
  """
  # Initialize generative model
  gamma = 0.9

  def feature_function(x):
    return x

  env = environment_factory('SIS', feature_function, **kwargs)

  # Evaluation limit parameters
  treatment_budget = np.int(np.floor(0.05 * kwargs['L']))
  evaluation_budget = treatment_budget

  random_policy = policy_factory('random')  # For initial actions
  argmaxer = argmaxer_factory(argmaxer_name)
  policy_arguments = {'treatment_budget': treatment_budget, 'env': env, 'divide_evenly': False}

  score_list = []
  for rep in range(n_rep):
    env.reset()
    env.step(random_policy(**policy_arguments)[0])
    env.step(random_policy(**policy_arguments)[0])
    for t in range(T-2):
      t0 = time.time()
      a, q_model = random_policy(**policy_arguments)
      # ToDo: update policy_arguments within policy
      policy_arguments['planning_depth'] = T - t
      policy_arguments['q_model'] = q_model
      env.step(a)
      t1 = time.time()
      print('rep: {} t: {} total time: {}'.format(rep, t, t1-t0))
    score_list.append(np.mean(env.Y))
    print('Episode score: {}'.format(np.mean(env.Y)))
  return score_list


if __name__ == '__main__':
  import time
  n_rep = 1
  SIS_kwargs = {'L': 50, 'omega': 0, 'generate_network': generate_network.lattice}
  k = 2
  scores = main(k, 25, n_rep, 'SIS', 'random', 'quad_approx', **SIS_kwargs)
  print('k={}: score={} se={}'.format(k, np.mean(scores), np.std(scores) / len(scores)))