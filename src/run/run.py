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

VALID_ENVIRONMENT_NAMES = ['sis', 'Ebola']
VALID_POLICY_NAMES = ['random', 'no_action', 'true_probs', 'true_probs_myopic', 'fqi', 'one_step',
                      'treat_all', 'SIS_stacked', 'SIS_model_based', 'sis_model_based_one_step',
                      'one_step_mse_averaged', 'sis_two_step_mse_averaged', 'sis_mb_fqi',
                      'ebola_model_based_one_step', 'ebola_model_based_myopic', 'policy_search',
                      'sis_one_step_equal_averaged', 'one_step_stacked']
VALID_ARGMAXER_NAMES = ['quad_approx', 'random', 'global', 'sequential_quad_approx']
VALID_NETWORK_NAMES = ['lattice', 'barabasi', 'nearestneighbor']

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', type=str, choices=VALID_ENVIRONMENT_NAMES)
  parser.add_argument('--policy_name', type=str, choices=VALID_POLICY_NAMES)
  parser.add_argument('--argmaxer_name', type=str, choices=VALID_ARGMAXER_NAMES)
  parser.add_argument('--number_of_replicates', type=int)
  parser.add_argument('--omega', type=float)
  parser.add_argument('--rollout_depth', type=int)
  parser.add_argument('--time_horizon', type=int)
  parser.add_argument('--L', type=int)
  parser.add_argument('--gamma', type=float)
  parser.add_argument('--evaluation_budget', type=int)
  parser.add_argument('--epsilon', type=float)
  parser.add_argument('--network', type=str, choices=VALID_NETWORK_NAMES)
  args = parser.parse_args()

  network_dict = {'lattice': generate_network.lattice, 'barabasi': generate_network.Barabasi_Albert,
                  'nearestneighbor': generate_network.random_nearest_neighbor}

  if args.env_name == 'sis':
    env_kwargs = {'L': args.L, 'omega': args.omega, 'generate_network': network_dict[args.network],
                  'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': args.epsilon}
    network_name = args.network
  else:
    env_kwargs = {}
    network_name = 'Ebola'
  Sim = Simulator(args.rollout_depth, args.env_name, args.time_horizon, args.number_of_replicates, args.policy_name,
                  args.argmaxer_name, args.gamma, args.evaluation_budget, env_kwargs, network_name)
  if args.number_of_replicates == 1:
    Sim.episode(0)
  else:
    Sim.run()
