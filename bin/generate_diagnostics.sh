#!/usr/bin/env bash
python3 ../src/run/generate_diagnostics.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
    --number_of_replicates=2 --omega=0.0 --rollout_depth=0 --time_horizon=5 --L=25 --gamma=0.9 \
    --num_bootstrap_samples=1 --diagnostic_type='compare_probability_estimates' --evaluation_budget=1

# python3 ../src/run/generate_diagnostics.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
#     --number_of_replicates=1 --omega=0.0 --rollout_depth=0 --time_horizon=25 --L=50 --gamma=0.9 \
#     --num_bootstrap_samples=1 --diagnostic_type='compare_probability_estimates'

exit
