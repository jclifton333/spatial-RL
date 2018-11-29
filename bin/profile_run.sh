#!/usr/bin/env bash
python3 -m cProfile -o profile_output ../src/run/run.py --env_name='Ebola' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=3 --L=20 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor'
cprofilev -f profile_output