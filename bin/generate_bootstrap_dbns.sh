#!/usr/bin/env bash
python3 ../src/run/generate_bootstrap_dbns.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
    --number_of_replicates=5 --omega=0.0 --rollout_depth=0 --time_horizon=25 --L=9 --gamma=0.9
