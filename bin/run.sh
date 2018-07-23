#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='SIS' --policy_name='rollout' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=5 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9 &

python3 ../src/run/run.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=5 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9 &

python3 ../src/run/run.py --env_name='SIS' --policy_name='rollout' --argmaxer_name='quad_approx' \
     --omega=1.0 --number_of_replicates=5 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9 &

python3 ../src/run/run.py --env_name='SIS' --policy_name='random' --argmaxer_name='quad_approx' \
     --omega=1.0 --number_of_replicates=5 --rollout_depth=0 --time_horizon=25 --L=100 --gamma=0.9
