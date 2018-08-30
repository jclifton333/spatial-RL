#!/usr/bin/env bash
# python3 ../src/run/run.py --env_name='sis' --policy_name='random' --argmaxer_name='quad_approx' \
#      --omega=0.5 --number_of_replicates=50 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0
# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='sis_model_based_one_step' --argmaxer_name='quad_approx' \
#      --omega=0.5 --number_of_replicates=50 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0

python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
     --omega=0.5 --number_of_replicates=50 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0

# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='true_probs' --argmaxer_name='quad_approx' \
#      --omega=1.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0

