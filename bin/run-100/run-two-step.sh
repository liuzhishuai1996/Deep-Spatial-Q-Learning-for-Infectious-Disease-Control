#!/bin/bash
#SBATCH -e run-two-step.err
#SBATCH --mem-per-cpu=2G
#SBATCH -c 20

module load CPLEX
module load Gurobi/8.11
################################# lattice + two_step + L=100 + GGCN/linear/liear_raw + epsilon=0 ################################# 
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
######
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
#####
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'

############################### nearestneighbor + two_step + L=100 + GGCN/linear/liear_raw + epsilon=0 ################################# 
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
######
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
######
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'

################################# contrived + two_step + L=100 + GGCN/linear/liear_raw + epsilon=0 ################################# 
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
######
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.5 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'
######
python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='True'

# python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='sweep' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=1 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='False' --save_features='False' --raw_features='False'