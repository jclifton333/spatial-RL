#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=2 --rollout_depth=1 --time_horizon=50 --L=10000 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=7 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False'

# python3 ../src/run/run.py --env_name='sis' --policy_name='random' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=4 --rollout_depth=1 --time_horizon=50 --L=10000 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False'
