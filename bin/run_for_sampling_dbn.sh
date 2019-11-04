#!/usr/bin/env bash
python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=25 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='true_probs_myopic'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=50 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='true_probs_myopic'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=100 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='true_probs_myopic'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=300 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='true_probs_myopic'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=25 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=50 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=100 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=300 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='lattice' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=25 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='nearestneighbor' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=50 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='nearestneighbor' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=100 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='nearestneighbor' --policy='random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=300 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
      --network='nearestneighbor' --policy='random'







