#!/bin/bash
###
 # @Description: 
 # @version: 
 # @Author: Zhishuai
 # @Date: 2023-01-08 17:50:41
### 
#SBATCH -e slurm.err
#SBATCH --mem-per-cpu=2G
#SBATCH -c 50

module load CPLEX
module load Gurobi/8.11
################################ lattice/nearestneighbor/contrived + policy search + L=100 + epsilon=0 ################################# 
python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'
#####
python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.5 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'
#####
python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='nearestneighbor' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='sis' --policy_name='policy_search' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=100 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=1 --network='contrived' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'