import numpy as np

# Hack bc python imports are stupid
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory
from src.utils.misc import KerasLogit, KerasRegressor
from src.estimation.stacking.bellman_error_bootstrappers import bootstrap_rollout_qfn


def run_sims_for_bootstrap_dbns(rollout_depth, num_bootstrap_samples, T, argmaxer_name, replicate, **kwargs):
  """
  :param rollout_depth:
  :param T: duration of simulation rep
  :param replicate: label for simulation replicate
  :param argmaxer_name: string in ['sweep', 'quad_approx'] for method of taking q function argmax
  :param kwargs: environment-specific keyword arguments
  """
  np.random.seed(replicate)

  # Initialize generative model
  gamma = 0.9

  def feature_function(x):
    return x

  env = environment_factory('SIS', feature_function, **kwargs)

  # Evaluation limit parameters
  # treatment_budget = np.int(np.floor(0.05 * kwargs['L']))
  treatment_budget = 1
  evaluation_budget = None

  random_policy = policy_factory('random')  # For initial actions
  argmaxer = argmaxer_factory(argmaxer_name)
  policy_arguments = {'treatment_budget': treatment_budget, 'env': env, 'divide_evenly': False}

  score_list = []
  times_to_save = [0, 1, 3, 5, 10, 15, 20, 30, 40, T-2]
  env.step(random_policy(**policy_arguments)[0])
  env.step(random_policy(**policy_arguments)[0])
  for t in range(T-2):
    a, q_model = random_policy(**policy_arguments)
    env.step(a)

    # Compute bootstrap BE
    if t in times_to_save:
      print('Computing bootstrap BE for time {}'.format(t))
      # mb_be = bootstrap_SIS_mb_qfn(env, KerasLogit, KerasRegressor, rollout_depth, gamma, T-t, q_model,
      #                              treatment_budget, evaluation_budget, argmaxer, num_bootstrap_samples)
      mf_be = bootstrap_rollout_qfn(env, KerasLogit, KerasRegressor, rollout_depth, gamma, treatment_budget,
                                    evaluation_budget, argmaxer, num_bootstrap_samples)
      # print('t: {}\nmb: {}\nmf: {}'.format(t, mb_be, mf_be))
      # bootstrap_results['mb_be'].append(mb_be)

  score_list.append(np.mean(env.Y))
  print('Episode score: {}'.format(np.mean(env.Y)))
  return score_list


if __name__ == '__main__':
  n_rep = 1
  omegas = [0, 0.5, 1]
  k = 0

  def mp_function(omega, replicate):
    SIS_kwargs = {'L': 9, 'omega': omega, 'generate_network': generate_network.lattice,
                  'initial_infections': np.random.binomial(1, p=0.3, size=9)}
    run_sims_for_bootstrap_dbns(k, 30, 3, 'global', replicate, **SIS_kwargs)
    return

  mp_function(0, 0)

