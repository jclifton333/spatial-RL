#!/usr/bin/env bash
python3 ../src/run/run_episode.py --env_name='Ebola' --policy_name='random' --argmaxer_name='sequential_quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=0 --time_horizon=25 --L=50 --gamma=0.9 \
     --evaluation_budget=3

