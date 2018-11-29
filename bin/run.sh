#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='Ebola' --policy_name='one_step_stacked' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=25 --L=20 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice'


