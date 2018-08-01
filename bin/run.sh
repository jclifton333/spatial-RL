#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5

python3 ../src/run/run.py --env_name='SIS' --policy_name='one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5

python3 ../src/run/run.py --env_name='SIS' --policy_name='SIS_model_based_one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5

python3 ../src/run/run.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0

python3 ../src/run/run.py --env_name='SIS' --policy_name='one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0

python3 ../src/run/run.py --env_name='SIS' --policy_name='SIS_model_based_one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0
