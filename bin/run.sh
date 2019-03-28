#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='true_probs' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=48 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=1


