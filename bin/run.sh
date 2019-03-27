#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=25 --L=30 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=1


