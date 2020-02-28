#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=7 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=8 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=9 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=10 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='lattice' --seed=7 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='lattice' --seed=8 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='lattice' --seed=9 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='lattice' --seed=10 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=7 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=8 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=9 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=10 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='nearestneighbor' --seed=7 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='nearestneighbor' --seed=8 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='nearestneighbor' --seed=9 --error_quantile=0.95 --ignore_errors='True'

python3 ../src/run/run.py --env_name='sis' --policy_name='sis_aic_two_step' --argmaxer_name='random' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=50 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.75 --network='nearestneighbor' --seed=10 --error_quantile=0.95 --ignore_errors='True'


