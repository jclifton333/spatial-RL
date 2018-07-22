#!/usr/bin/env bash
python3 ../src/run/generate_bootstrap_dbns.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
    --number_of_replicates=2 --omega=0.0 --rollout_depth=0 --time_horizon=3 --L=9 --gamma=0.9 \
    --num_bootstrap_samples=1

# python3 ../src/run/generate_bootstrap_dbns.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
#     --number_of_replicates=1 --omega=0.5 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9 \
#     --num_bootstrap_samples=30 &
#
# python3 ../src/run/generate_bootstrap_dbns.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
#     --number_of_replicates=1 --omega=1.0 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9 \
#     --num_bootstrap_samples=30
