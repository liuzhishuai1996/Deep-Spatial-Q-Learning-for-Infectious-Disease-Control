#!/usr/bin/env bash

#SBATCH --mem=32G
#SBATCH -c 20
#module load Gurobi/8.11

python3 ./src/run/run.py --env_name='sis' --policy_name='two_step' --argmaxer_name='quad_approx' \
     --omega=0.0 --number_of_replicates=2 --rollout_depth=1 --time_horizon=25 --L=100 --gamma=0.9 \
     --evaluation_budget=100 --epsilon=0 --network='lattice' --seed=5 --error_quantile=0.95 --ignore_errors='True' \
     --learn_embedding='True' --save_features='False' --raw_features='False'