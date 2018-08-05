#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='SIS' --policy_name='sis_one_step_be_averaged' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=40 --rollout_depth=1 --time_horizon=25 --L=50 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0

