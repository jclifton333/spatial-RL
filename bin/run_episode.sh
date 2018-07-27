#!/usr/bin/env bash
python3 ../src/run/run_episode.py --env_name='SIS' --policy_name='rollout' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=4 --L=9 --gamma=0.9 \
     --evaluation_budget=3

