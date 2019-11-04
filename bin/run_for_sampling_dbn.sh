#!/usr/bin/env bash
python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=300 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True'
