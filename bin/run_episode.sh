#!/usr/bin/env bash
python3 ../src/run/run_episode.py --env_name='SIS' --policy_name='one_step' \
     --argmaxer_name='quad_approx' --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=1000 --L=100 --gamma=0.9 \
     --evaluation_budget=3 --epsilon=0.0

