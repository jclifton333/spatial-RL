#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='two_step_ggcn' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

python3 ../src/run/run.py --env_name='sis' --policy_name='two_step_ggcn' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

python3 ../src/run/run.py --env_name='sis' --policy_name='two_step_ggcn' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

python3 ../src/run/run.py --env_name='sis' --policy_name='two_step_ggcn' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='oracle_multiple_quad_approx' \
#      --omega=0.0 --number_of_replicates=12 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ../src/run/run.py --env_name='Ebola' --policy_name='one_step' --argmaxer_name='multiple_quad_approx' \
#      --omega=0.0 --number_of_replicates=12 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=4 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'

# python3 ../src/run/run.py --env_name='Ebola' --policy_name='one_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=12 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=4 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'
# 
# python3 ../src/run/run.py --env_name='Ebola' --policy_name='one_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=12 --rollout_depth=1 --time_horizon=25 --L=300 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=4 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'

# python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=4 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'
# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=24 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=4 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

