#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='sis_one_step_dyna_space_filling' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=5 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=7 --error_quantile=0.95
