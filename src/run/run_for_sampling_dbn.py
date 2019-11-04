# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import argparse

# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.run.Simulator import Simulator

POLICY_NAME = 'true_probs_myopic'
ENV_NAME = 'sis'
NETWORK = 'lattice'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--number_of_replicates', type=int)
  parser.add_argument('--omega', type=float)
  parser.add_argument('--rollout_depth', type=int)
  parser.add_argument('--time_horizon', type=int)
  parser.add_argument('--L', type=int)
  parser.add_argument('--gamma', type=float)
  parser.add_argument('--evaluation_budget', type=int)
  parser.add_argument('--epsilon', type=float)
  parser.add_argument('--ts', type=str, choices=['True', 'False'])
  parser.add_argument('--seed', type=int)
  parser.add_argument('--num_prefit_data', type=float)
  parser.add_argument('--error_quantile', type=float)
  parser.add_argument('--ignore_errors', type=str)
  args = parser.parse_args()

  env_kwargs = {'L': args.L, 'omega': args.omega, 'generate_network': generate_network.random_nearest_neighbor,
                'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': args.epsilon}

  ts = (args.ts == 'True')
  ignore_errors = (args.ignore_errors == 'True')

  Sim = Simulator(args.rollout_depth, ENV_NAME, args.time_horizon, args.number_of_replicates, POLICY_NAME,
                  'random', args.gamma, args.evaluation_budget, env_kwargs, NETWORK, ts, args.seed,
                  args.error_quantile, fit_qfn_at_end=True, sampling_dbn_run=True, ignore_errors=ignore_errors)
  if args.number_of_replicates == 1:
    Sim.episode(0)
  else:
    Sim.run_for_sampling_dbn()

