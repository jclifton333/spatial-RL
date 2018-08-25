#!/usr/bin/env bash
python3 -m cProfile -o profile_output ../src/run/run_episode.py --env_name='sis' --policy_name='fqi' \
     --argmaxer_name='sequential_quad_approx' \
     --omega=0.0 --number_of_replicates=1 --rollout_depth=0 --time_horizon=25 --L=50 --gamma=0.9 \
     --evaluation_budget=20
cprofilev -f profile_output