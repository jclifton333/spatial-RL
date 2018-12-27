#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='two_step_sis_prefit' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=16 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=1


