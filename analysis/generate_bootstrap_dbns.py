import numpy as np
import pdb

# Hack bc python imports are stupid
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.estimation.q_functions.rollout import rollout
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.stacking.compute_sample_bellman_error import compute_sample_squared_bellman_error
from src.policies.policy_factory import policy_factory
from src.utils.misc import KerasLogit, KerasRegressor, SKLogit
from analysis.bellman_error_bootstrappers import bootstrap_SIS_mb_qfn, bootstrap_rollout_qfn
import pickle as pkl


def run_sims_for_bootstrap_dbns(rollout_depth, num_bootstrap_samples, T, n_rep, argmaxer_name, **kwargs):
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

  fname = 'bootstrap-dbns-omega-{}-horizon-{}.p'.format(kwargs['omega'], T)
  bootstrap_results = {'time': [], 'mb_be': [], 'mf_be': [], 'omega': kwargs['omega'],
                       'notes': 'number of simulation reps: 50'}

  score_list = []
  times_to_save = [0, 10, 20, 30, 40, 45]
  for rep in range(n_rep):
    env.reset()
    env.step(random_policy(**policy_arguments)[0])
    env.step(random_policy(**policy_arguments)[0])
    for t in range(T-2):
      a, q_model = random_policy(**policy_arguments)
      env.step(a)

      # Compute bootstrap BE
      if t in times_to_save:
        print('Computing bootstrap BE for time {}'.format(t))
        mb_be = bootstrap_SIS_mb_qfn(env, KerasLogit, KerasRegressor, rollout_depth, gamma, T-t, q_model,
                                     treatment_budget, evaluation_budget, argmaxer, num_bootstrap_samples)
        mf_be = bootstrap_rollout_qfn(env, SKLogit, KerasRegressor, rollout_depth, gamma, treatment_budget,
                                      evaluation_budget, argmaxer, num_bootstrap_samples)
        print('t: {}\nmb: {}\nmf: {}'.format(t, mb_be, mf_be))
        bootstrap_results['time'].append(t)
        bootstrap_results['mb_be'].append(mb_be)
        bootstrap_results['mf_be'].append(mf_be)
        pkl.dump(bootstrap_results, open(fname, 'wb'))

    score_list.append(np.mean(env.Y))
    print('Episode score: {}'.format(np.mean(env.Y)))
  return score_list


if __name__ == '__main__':
  import time
  n_rep = 1
  omegas = [0, 0.5, 1]
  k = 0
  for omega in omegas:
    SIS_kwargs = {'L': 50, 'omega': omega, 'generate_network': generate_network.lattice}
    run_sims_for_bootstrap_dbns(k, 30, 50, n_rep, 'quad_approx', **SIS_kwargs)
