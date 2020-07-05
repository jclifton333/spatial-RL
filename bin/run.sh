#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='nonlinear' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=50 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=7 --error_quantile=0.95 --ignore_errors='True'
