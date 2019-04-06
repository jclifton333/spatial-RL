#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=2 --error_quantile=0.95

python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='lattice' --seed=2 --error_quantile=0.95

python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1.0 --network='lattice' --seed=2 --error_quantile=0.95
