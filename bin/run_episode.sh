#!/usr/bin/env bash
python3 ../src/run/run_episode.py --env_name='SIS' --policy_name='dummy_stacked' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=0 --time_horizon=3 --L=30 --gamma=0.9 \
     --evaluation_budget=3

