#!/usr/bin/env bash
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=25 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='lattice' --policy='true_probs_myopic' --sampling_dbn_estimator='two_step_random'

# py ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='lattice' --policy='true_probs_myopic' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='nearestneighbor' --policy='true_probs_myopic' --sampling_dbn_estimator='two_step_random'
# 
# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#       --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='nearestneighbor' --policy='true_probs_myopic' --sampling_dbn_estimator='two_step_random'

# python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
#      --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='True' \
#       --network='lattice' --policy='true_probs_myopic' --sampling_dbn_estimator='two_step_random'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=1 --rollout_depth=1 --time_horizon=25 \
      --L=100 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=100 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=500 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=1000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=5000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='nearestneighbor' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'

python3 ../src/run/run_for_sampling_dbn.py  --omega=0.0 --number_of_replicates=480 --rollout_depth=1 --time_horizon=25 \
      --L=5000 --gamma=0.9 --evaluation_budget=100 --epsilon=0.0 --seed=7 --error_quantile=0.95 --ignore_errors='False' \
      --network='lattice' --policy='random' --sampling_dbn_estimator='two_step_mb_constant_cutoff'










