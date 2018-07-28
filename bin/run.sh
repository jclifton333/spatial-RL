#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=50 --rollout_depth=1 --time_horizon=25 --L=50 --gamma=0.9 \
     --evaluation_budget=100


