#!/usr/bin/env bash
python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=2 --rollout_depth=1 --time_horizon=10 --L=30 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=1

# python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='barabasi' --seed=1
# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=1

# python3 ../src/run/run.py --env_name='sis' --policy_name='one_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=1
# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='barabasi' --seed=1
# 
# python3 ../src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=48 --rollout_depth=1 --time_horizon=50 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=1
# 
