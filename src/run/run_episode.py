
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

VALID_ENVIRONMENT_NAMES = ['SIS', 'Ebola']
VALID_POLICY_NAMES = ['random', 'no_action', 'true_probs', 'true_probs_myopic', 'rollout', 'rollout', 'one_step',
                      'treat_all', 'SIS_stacked', 'SIS_model_based', 'SIS_model_based_one_step']
VALID_ARGMAXER_NAMES = ['quad_approx', 'sequential_quad_approx', 'random', 'global']

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
  args = parser.parse_args()

  SIS_kwargs = {'L': args.L, 'omega': args.omega, 'generate_network': generate_network.lattice,
                'initial_infections': None, 'add_neighbor_sums': True}
  Sim = Simulator(args.rollout_depth, args.env_name, args.time_horizon, args.number_of_replicates, args.policy_name,
                  args.argmaxer_name, args.gamma, args.evaluation_budget, **SIS_kwargs)
  Sim.episode(0)