# -*- coding: utf-8 -*-N
"""
Created on Fri May  4 21:49:40 2018

@author: Jesse
"""
import numpy as np
import time
import datetime
import yaml
import torch.multiprocessing as mp

import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

from src.environments.environment_factory import environment_factory
from src.estimation.optim.argmaxer_factory import argmaxer_factory
from src.policies.policy_factory import policy_factory
import copy

from src.estimation.q_functions.model_fitters import SKLogit2
from sklearn.linear_model import Ridge

class Simulator(object):
  def __init__(self, lookahead_depth, env_name, time_horizon, number_of_replicates, policy_name, argmaxer_name, gamma,
               evaluation_budget, env_kwargs, network_name, bootstrap, seed, error_quantile,
               sampling_dbn_run=False, sampling_dbn_estimator=None, fit_qfn_at_end=False, variance_only=False,
               parametric_bootstrap=False, ignore_errors=False, fname_addendum=None, save_features=False,
               raw_features=False, diagnostic_mode=False):

    self.env_name = env_name
    self.env_kwargs = env_kwargs
    self.env = environment_factory(env_name, **env_kwargs) 
    self.policy = policy_factory(policy_name)
    self.random_policy = policy_factory('random') 
    self.argmaxer = argmaxer_factory(argmaxer_name)
    self.time_horizon = time_horizon
    self.number_of_replicates = number_of_replicates
    self.ignore_errors = ignore_errors 
    self.scores = []
    self.runtimes = []
    self.seed = seed
    self.fit_qfn_at_end = fit_qfn_at_end
    self.sampling_dbn_estimator = sampling_dbn_estimator
    self.variance_only = variance_only
    self.parametric_bootstrap = parametric_bootstrap
    self.save_features = save_features
    self.raw_features = raw_features
    self.gamma = gamma

    # Set policy arguments
    if env_name in ['sis', 'ContinuousGrav']:
        treatment_budget = np.int(np.ceil(0.05 * self.env.L))
    elif env_name == 'Ebola':
        treatment_budget = np.int(np.ceil(0.15 * self.env.L))
    self.policy_arguments = {'classifier': SKLogit2, 'regressor': Ridge, 'env': self.env,
                              'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                              'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                              'divide_evenly': False, 'argmaxer': self.argmaxer, 'q_model': None,
                              'bootstrap': bootstrap, 'initial_policy_parameter': None, 'q_fn': None,
                              'quantile': error_quantile, 'raw_features': self.raw_features,
                              'diagnostic_mode': diagnostic_mode}

    # Get settings dict for log
    if 'epsilon' in env_kwargs.keys():
        epsilon = env_kwargs['epsilon']
    else:
        epsilon = None
    self.settings = {'classifier': self.policy_arguments['classifier'].__name__,
                     'regressor': self.policy_arguments['regressor'].__name__,
                     'evaluation_budget': evaluation_budget, 'gamma': gamma, 'rollout_depth': lookahead_depth,
                     'planning_depth': self.time_horizon, 'treatment_budget': treatment_budget,
                     'divide_evenly': self.policy_arguments['divide_evenly'], 'argmaxer': argmaxer_name,
                     'evaluation_policy': self.sampling_dbn_estimator, 'epsilon': epsilon}
    self.settings.update({'env_name': env_name, 'L': self.env.L, 'policy_name': policy_name,
                          'argmaxer_name': argmaxer_name, 'time_horizon': self.time_horizon,
                          'number_of_replicates': self.number_of_replicates,
                          'learn_embedding': str(self.env.learn_embedding),
                          'raw_features': str(self.raw_features)})

    # Get filename base for saving results
    to_join = [env_name, policy_name, argmaxer_name, str(self.env.L), network_name, 'learn-embedding={}'.format(self.env.learn_embedding), 'raw-feature={}'.format(self.raw_features),
               'eval-policy={}'.format(self.sampling_dbn_estimator)]
    if sampling_dbn_run:
      to_join.append('eval={}'.format(self.sampling_dbn_estimator))
    if 'epsilon' in env_kwargs.keys():
      to_join.append(str(env_kwargs['epsilon']))
    if fname_addendum is not None:
      to_join.append(fname_addendum)
    self.basename = '_'.join(to_join)

  def run(self):
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 10))
    pool = mp.Pool(processes=num_processes)

    if self.ignore_errors:
      results_list = pool.map(self.episode_wrapper, [i for i in range(self.number_of_replicates)])
    else:
      results_list = pool.map(self.episode, [i for i in range(self.number_of_replicates)])

    results_dict = {k: v for d in results_list for k, v in d.items() if d is not None}
    self.save_results(results_dict)

    return

  def episode_wrapper(self, replicate):
    return self.episode(replicate)

  def episode(self, replicate):
    np.random.seed(int(self.seed*self.number_of_replicates + replicate))
    episode_results = {'score': None, 'runtime': None}
    t0 = time.time()
    self.env.reset()

    # Initial steps 
    self.env.step(self.random_policy(**self.policy_arguments)[0]) 
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    mean_infection_rate = []
    linear_acc_lst = []
    gccn_acc_lst = []
    q_diff_lst = []
    for t in range(self.time_horizon-5):
      print(t)
      tt0 = time.time()
      a, info = self.policy(**self.policy_arguments) 
      self.policy_arguments['planning_depth'] = self.time_horizon - t 
      print(self.env.Y[-1, :].mean())

      mean_infection = float(self.env.Y[-1, :].mean())
      mean_infection_rate.append(mean_infection)

      self.env.step(a) 
      tt1 = time.time()
      print(float(tt1-tt0))


    t1 = time.time()
    score = np.mean(self.env.Y) 
    discounted_factor = [np.power(self.gamma,i) for i in range(self.env.Y.shape[0])]
    V = np.dot(np.sum(self.env.Y,1), discounted_factor)
    # score = np.mean(self.env.current_infected)
    episode_results['score'] = float(score)
    episode_results['V'] = float(V)
    episode_results['runtime'] = float(t1 - t0)
    episode_results['mean_infection_rate'] = mean_infection_rate

    if len(linear_acc_lst) > 0:
      episode_results['linear_acc_lst'] = linear_acc_lst
      episode_results['gccn_acc_lst'] = gccn_acc_lst

    if self.fit_qfn_at_end:
      # Get q-function parameters
      q_fn_policy_params = copy.deepcopy(self.policy_arguments)
      q_fn_policy_params['rollout'] = False
      q_fn_policy_params['rollout_env'] = None
      q_fn_policy_params['rollout_policy'] = None
      q_fn_policy_params['time_horizon'] = self.time_horizon
      q_fn_policy_params['classifier'] = SKLogit2
      q_fn_policy_params['regressor'] = Ridge
      q_fn_policy_params['bootstrap'] = False
      q_fn_policy = policy_factory(self.sampling_dbn_estimator)
      _, q_fn_policy_info = q_fn_policy(**q_fn_policy_params)
      episode_results['q_fn_params'] = [float(t) for t in q_fn_policy_info['q_fn_params']]
      episode_results['q_fn_params_raw'] = [float(t) for t in q_fn_policy_info['q_fn_params_raw']]

    print(f'score: {score}')
    print(f'V: {V}')

    if self.save_features and self.number_of_replicates == 1: 
      X_raw = self.env.X_raw
      y = self.env.y
      adjacency_mat = self.env.adjacency_matrix
      save_dict = {'X_raw': X_raw, 'y': y, 'adjacency_mat': adjacency_mat}
      prefix = os.path.join(pkg_dir, 'analysis', 'observations', self.basename)
      suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
      filename = '{}_{}.npy'.format(prefix, suffix)
      np.save(filename, save_dict)

    return {replicate: episode_results}

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return
