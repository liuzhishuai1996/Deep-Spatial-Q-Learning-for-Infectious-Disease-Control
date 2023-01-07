# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import argparse
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

from src.environments import generate_network
from src.run.Simulator import Simulator
from src.run.test import *
import torch
import faulthandler

VALID_ENVIRONMENT_NAMES = ['sis', 'Ebola', 'ContinuousGrav']
VALID_POLICY_NAMES = ['random', 'no_action', 'true_probs', 'true_probs_myopic', 'fqi', 'one_step', 'two_step', 'three_step',
                      'treat_all', 'sis_model_based_one_step', 'two_step_mb', 'two_step_gnn', 'three_step_gnn', 'one_step_gnn',
                      'one_step_mse_averaged', 'sis_two_step_mse_averaged',
                      'gravity_model_based_one_step', 'gravity_model_based_myopic', 'policy_search',
                      'sis_one_step_equal_averaged', 'one_step_stacked', 'sis_model_based_myopic',
                      'two_step_higher_order', 'two_step_sis_prefit', 'one_step_truth_augmented',
                      'one_step_projection_combo', 'two_step_stacked', 'sis_aic_two_step', 'sis_aic_one_step',
                      'sis_one_step_continuation', 'ebola_aic_one_step', 'sis_one_step_dyna_space_filling',
                      'sis_local_aic_one_step', 'ebola_aic_two_step', 'two_step_oracle_ggcn', 'two_step_ggcn',
                      'oracle_policy_search', 'two_step_true_probs']
POLICY_SEARCH_NAMES = ['policy_search', 'sis_aic_two_step', 'sis_one_step_continuation', 'ebola_aic_two_step',
                       'oracle_policy_search']
VALID_ARGMAXER_NAMES = ['quad_approx', 'random', 'global', 'sequential_quad_approx', 'nonlinear', 'searching',
                        'multiple_quad_approx', 'oracle_multiple_quad_approx']
VALID_NETWORK_NAMES = ['lattice', 'barabasi', 'nearestneighbor', 'contrived']

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', type=str, choices=VALID_ENVIRONMENT_NAMES)
  parser.add_argument('--policy_name', type=str, choices=VALID_POLICY_NAMES)
  parser.add_argument('--argmaxer_name', type=str, choices=VALID_ARGMAXER_NAMES)
  parser.add_argument('--number_of_replicates', type=int)
  parser.add_argument('--omega', type=float)
  parser.add_argument('--rollout_depth', type=int)
  parser.add_argument('--time_horizon', type=int)
  parser.add_argument('--L', type=int)
  parser.add_argument('--gamma', type=float)
  parser.add_argument('--evaluation_budget', type=int)
  parser.add_argument('--epsilon', type=float)
  parser.add_argument('--network', type=str, choices=VALID_NETWORK_NAMES)
  parser.add_argument('--ts', type=str, choices=['True', 'False'])
  parser.add_argument('--seed', type=int)
  parser.add_argument('--num_prefit_data', type=float)
  parser.add_argument('--error_quantile', type=float)
  parser.add_argument('--ignore_errors', type=str)
  parser.add_argument('--learn_embedding', type=str, choices=['True', 'False'])
  parser.add_argument('--save_features', type=str, choices=['True', 'False'])
  parser.add_argument('--raw_features', type=str, choices=['True', 'False'], default='False')
  parser.add_argument('--diagnostic_mode', type=str, choices=['True', 'False'], default='False')
  #args = parser.parse_args()

  class get_args():
    def __init__(self, env_name, policy_name, argmaxer_name, seed, rollout_depth, number_of_replicates, omega, 
                 time_horizon, evaluation_budget, L, gamma, epsilon, network, learn_embedding, raw_features, 
                 error_quantile, ignore_errors, save_features, ts, diagnostic_mode):
      self.env_name = env_name
      self.policy_name = policy_name
      self.argmaxer_name = argmaxer_name
      self.seed = seed
      self.number_of_replicates = number_of_replicates
      self.time_horizon = time_horizon
      self.evaluation_budget = evaluation_budget
      self.L = L
      self.gamma = gamma
      self.rollout_depth = rollout_depth
      self.omega = omega
      self.epsilon = epsilon
      self.network = network
      self.learn_embedding = learn_embedding
      self.raw_features = raw_features
      self.error_quantile = error_quantile
      self.ignore_errors = ignore_errors
      self.save_features = save_features
      self.ts = ts
      self.diagnostic_mode = diagnostic_mode
  args = get_args(env_name = 'sis', policy_name = 'three_step', argmaxer_name = 'quad_approx', seed = 60, rollout_depth = 1, number_of_replicates = 1,
                  omega = 0, time_horizon = 25, evaluation_budget = 100, L = 100, gamma = 0.9, epsilon = 0, network = 'lattice',
                  learn_embedding = 'True', raw_features = 'True', error_quantile = 0.95, ignore_errors = 'True', save_features = 'False',
                  ts = 'True', diagnostic_mode = 'False')


  network_dict = {'lattice': generate_network.lattice, 'nearestneighbor': generate_network.random_nearest_neighbor,
                  'contrived': generate_network.contrived}

  if args.env_name == 'sis':
    env_kwargs = {'L': args.L, 'omega': args.omega, 'generate_network': network_dict[args.network],
                  'initial_infections': None, 'add_neighbor_sums': False, 'epsilon': args.epsilon,
                  'learn_embedding': (args.learn_embedding == 'True')}
    if args.network == 'nearestneighbor':
      env_kwargs['regenerate_network'] = True
    network_name = args.network
  elif args.env_name == 'Ebola':
    env_kwargs = {'learn_embedding': (args.learn_embedding == 'True')}
    network_name = 'Ebola'
  elif args.env_name == 'ContinuousGrav':
    env_kwargs = {'L': args.L}
    network_name = 'ContinuousGrav'
    
  ts = (args.ts == 'True') #multiplier bootstrap
  ignore_errors = (args.ignore_errors == 'True')

  if args.policy_name in POLICY_SEARCH_NAMES:
    env_kwargs['construct_features_for_policy_search'] = True
    if 'sis' in args.policy_name:
      env_kwargs['neighbor_features'] = False

  faulthandler.enable()
  Sim = Simulator_treat2(args.rollout_depth, args.env_name, args.time_horizon, args.number_of_replicates, args.policy_name,
                  args.argmaxer_name, args.gamma, args.evaluation_budget, env_kwargs, network_name, ts, args.seed,
                  args.error_quantile, save_features=(args.save_features == 'True'), ignore_errors=ignore_errors,
                  raw_features=(args.raw_features == 'True'), diagnostic_mode=(args.diagnostic_mode == 'True'))
  if args.number_of_replicates == 1:
    Sim.episode(0)
  else:
    if args.save_features == 'True':
      raise ValueError("Must have number_of_replicates=1 if save_features==True")
    else:
      torch.multiprocessing.set_start_method('forkserver', force = True)
      Sim.run()
