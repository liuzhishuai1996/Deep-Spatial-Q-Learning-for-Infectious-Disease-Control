#!/bin/bash
###
 # @Description: 
 # @version: 
 # @Author: Zhishuai
 # @Date: 2022-09-13 21:08:30
### 
#SBATCH -e run-Ebola-three-step.err
#SBATCH --mem-per-cpu=2G
#SBATCH -c 20

module load CPLEX
module load Gurobi/8.11
################################# lattice + three_step + L=100 + GGCN/linear/liear_raw + epsilon=0 ################################# 
# python3 ./src/run/run.py --env_name='Ebola' --policy_name='three_step' --argmaxer_name='quad_approx' \
#      --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
#      --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
#      --learn_embedding='True' --save_features='False' --raw_features='False'

python3 ./src/run/run.py --env_name='Ebola' --policy_name='three_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='True'

python3 ./src/run/run.py --env_name='Ebola' --policy_name='three_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=20 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0.0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='False' --save_features='False' --raw_features='False'
