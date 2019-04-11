#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='one_step_stacked' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=2 --error_quantile=0.95

python3 ../src/run/run.py --env_name='sis' --policy_name='one_step_stacked' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=2 --error_quantile=0.95

python3 ../src/run/run.py --env_name='sis' --policy_name='one_step_stacked' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='nearestneighbor' --seed=2 --error_quantile=0.95

python3 ../src/run/run.py --env_name='sis' --policy_name='one_step_stacked' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1.0 --network='nearestneighbor' --seed=2 --error_quantile=0.95



