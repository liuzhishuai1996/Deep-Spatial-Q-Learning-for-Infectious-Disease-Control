# -*- coding: utf-8 -*-N

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
from src.estimation.q_functions.embedding import oracle_tune_ggcn, learn_ggcn
from src.estimation.q_functions.embedding1 import oracle_tune_ggcn1, learn_ggcn1
from src.estimation.q_functions.embedding2 import learn_ggcn2
from src.estimation.q_functions.one_step import *
from sklearn.linear_model import LogisticRegression
from src.utils.misc import kl


class Simulator_test(object):
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
    num_processes = np.min((self.number_of_replicates, 6))
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
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    for t in range(self.time_horizon-2):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None
      #GNN
      #_, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list, neighbor_order=1)
      _, predictor = learn_ggcn2(env.X_raw, env.y, env.adjacency_list, n_epoch=200, binary = True)

      #Logistic regression
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X)
      clf.fit(X, y)
      print(self.env.Y[-1, :].mean())

      def oracle_qfn(a_):
        return env.next_infected_probabilities(a_) 
      def qfn_gnn(a_):
        X_ = env.data_block_at_action(-1, a_, raw=True)
        return predictor(X_)
      def qfn_l(a_):
        X_ = env.data_block_at_action(-1, a_, neighbor_order=1)
        return clf.predict_proba(X_)[:, 1]


      #compute absolute error and kl distance
      N_REP = 100 #从均匀分布中随机产生N_REP组治疗
      dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
      eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)] 

      linear_error = 0.
      gnn_error = 0.
      linear_distance = 0.
      gnn_distance = 0.
      base_error = 0.
      base_distance = 0.
      for a_ in eval_actions:
        gnn_probs = qfn_gnn(a_)
        true_probs = oracle_qfn(a_)
        linear_probs = qfn_l(a_)
        linear_error += round(abs(np.array(linear_probs - true_probs)).sum(), 2) / N_REP
        gnn_error += round(abs(np.array(gnn_probs - true_probs)).sum(), 2) / N_REP
        linear_distance += round(kl(linear_probs, true_probs), 5) / N_REP
        gnn_distance += round(kl(gnn_probs, true_probs), 5) / N_REP
        base_error += round(abs(np.array(true_probs)).sum(), 2) / N_REP
        base_distance += round(kl(np.zeros(len(true_probs)), true_probs), 5) / N_REP
      print(f'gccn error: {gnn_error} linear error: {linear_error} base error: {base_error}')
      print(f'gccn distance: {gnn_distance} linear distance: {linear_distance} base distance: {base_distance}')

      self.env.step(self.random_policy(**self.policy_arguments)[0])

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

class Simulator_test1(object):
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
    num_processes = np.min((self.number_of_replicates, 6))
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

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return
    
  def episode(self, replicate):
    np.random.seed(int(self.seed*self.number_of_replicates + replicate))
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    Mse_gnn = []
    Mse_l = []
    Base = []

    for t in range(self.time_horizon-5):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None
      #Logistic
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X)
      clf.fit(X, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a))[:,1]
      
      #Back up once
      backup = []
      for t in range(1, env.T):
        qfn_at_block_t = lambda a: qfn_at_block(t, a)
        a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
        q_max = qfn_at_block_t(a_max)
        backup_at_t = q_max
        backup.append(backup_at_t)

      reg = regressor()
      X = np.vstack(env.X[3:-1])
      y = np.hstack(env.y[3:-1])
      reg.fit(X, y + gamma*np.hstack(backup[3:]))
      
      y = []
      for i in range(len(backup[3:])):
        y.append(env.y[3:-1][i] + gamma * backup[3+i])
      #_, predictor1 = learn_ggcn1(env.X_raw[3:-1], y, env.adjacency_list, neighbor_order=2)
      _, predictor1 = learn_ggcn2(env.X_raw[3:-1], y, env.adjacency_list, binary = False)
      #testing
      mse_gnn = 0.
      mse_l = 0.
      base = 0.
      for t in range(3):
         y_hat_gnn = predictor1(env.X_raw[t])
         y_hat_l = reg.predict(env.X[t])
         y = env.y[t] + backup[t]
         mse_gnn += ((y_hat_gnn - y)**2).mean() / 3
         mse_l += ((y_hat_l - y)**2).mean() / 3
         base += (y**2).mean() / 3
        
      Mse_gnn.append(mse_gnn)
      Mse_l.append(mse_l)
      Base.append(base)

      print(f'mse_gnn: {mse_gnn} acc_l: {mse_l} base: {base}')
      print(self.env.Y[-1, :].mean())


      self.env.step(self.random_policy(**self.policy_arguments)[0])
    episode_results['mse_gnn'] = Mse_gnn
    episode_results['mse_hc'] = Mse_l
    episode_results['base'] = Base
    
    return {replicate: episode_results}

class Simulator_test2(object):
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
    num_processes = np.min((self.number_of_replicates, 6))
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

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return
    
  def episode(self, replicate):
    np.random.seed(int(self.seed*self.number_of_replicates + replicate))
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    Mse_gnn = []
    Mse_l = []
    Base1 = []
    Base2 = []

    for t in range(self.time_horizon-5):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None
      #Logistic
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X = np.vstack(env.X)
      clf.fit(X, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a))[:,1]
      
      #Back up once
      backup = []
      for t in range(1, env.T):
        qfn_at_block_t = lambda a: qfn_at_block(t, a)
        a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
        q_max = qfn_at_block_t(a_max)
        backup_at_t = q_max
        backup.append(backup_at_t)

      reg = regressor()
      X = np.vstack(env.X_2[3:-1])
      y = np.hstack(env.y[3:-1])
      reg.fit(X, y + gamma*np.hstack(backup[3:]))
      
      y = []
      for i in range(len(backup[3:])):
        y.append(env.y[3:-1][i] + gamma * backup[3+i])
      _, predictor1 = learn_ggcn1(env.X_raw[3:-1], y,
                                  env.adjacency_list, neighbor_order=2)
      
      def qfn_at_block_1(block_index, a):
        return predictor1(env.data_block_at_action(block_index, a, raw = True))

      def qfn_at_block_1_l(block_index, a):
        return reg.predict(env.data_block_at_action(block_index, a, neighbor_order = 2))

      backup1 = []
      backup2 = []
      for t in range(1, env.T):
        qfn_at_block_t = lambda a: qfn_at_block_1(t, a)
        a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
        q_max = qfn_at_block_t(a_max)
        backup1_at_t = q_max
        backup1.append(backup1_at_t)

        qfn_at_block_t = lambda a: qfn_at_block_1_l(t, a)
        a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
        q_max = qfn_at_block_t(a_max)
        backup2_at_t = q_max
        backup2.append(backup2_at_t)

      reg1 = regressor()
      X = np.vstack(env.X_2[3:-1])
      y = np.hstack(env.y[3:-1])
      reg1.fit(X, y + gamma*np.hstack(backup2[3:]))
      
      y = []
      for i in range(len(backup1[3:])):
        y.append(env.y[3:-1][i] + gamma * backup1[3+i])
      _, predictor2 = learn_ggcn1(env.X_raw[3:-1], y,
                                  env.adjacency_list, neighbor_order=1)

      #testing
      mse_gnn = 0.
      mse_l = 0.
      base1 = 0.
      base2 = 0.
      for t in range(3):
         y_hat_gnn = predictor2(env.X_raw[t])
         y_hat_l = reg1.predict(env.X_2[t])
         y1 = env.y[t] + backup1[t]
         mse_gnn += ((y_hat_gnn - y1)**2).mean() / 3
         y2 = env.y[t] + backup2[t]
         mse_l += ((y_hat_l - y2)**2).mean() / 3
         base1 += (y1**2).mean() / 3
         base2 += (y2**2).mean() / 3
        
      Mse_gnn.append(mse_gnn)
      Mse_l.append(mse_l)
      Base1.append(base1)
      Base2.append(base2)


      print(f'mse_gnn: {mse_gnn} acc_l: {mse_l} base1: {base1} base2: {base2}')
      print(self.env.Y[-1, :].mean())


      self.env.step(self.random_policy(**self.policy_arguments)[0])
    episode_results['mse_gnn'] = Mse_gnn
    episode_results['mse_hc'] = Mse_l
    episode_results['base1'] = Base1
    episode_results['base2'] = Base2
    
    return {replicate: episode_results}

class Simulator_treat(object):
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

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return

  def run(self):
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 6))
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
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    Acc_gnn = []
    Acc_l = []

    for t in range(self.time_horizon-5):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None 
      
      #GNN
      _, predictor = learn_ggcn(env.X_raw[3:], env.y[3:], env.adjacency_list, neighbor_order=1)
      
      #Logistic regression
      #clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      clf = LogisticRegression()
      y = np.hstack(env.y[3:])
      X = np.vstack(env.X_2[3:])
      clf.fit(X, y)

      #testing
      acc_gnn = 0.
      acc_l = 0.
      for t in range(3):
         y_hat_gnn = (predictor(env.X_raw[t]) > 0.5)
         #y_hat_l = (clf.predict_proba(env.X[t], **predict_proba_kwargs) > 0.5)
         y_hat_l = (clf.predict_proba(env.X_2[t])[:, 1] > 0.5)
         acc_gnn += (y_hat_gnn == env.y[t]).mean() / 3
         acc_l += (y_hat_l == env.y[t]).mean() / 3
        
      Acc_gnn.append(acc_gnn)
      Acc_l.append(acc_l)

      print(f'acc_gnn: {acc_gnn} acc_l: {acc_l}')
      print(self.env.Y[-1, :].mean())
      

      def oracle_qfn(a_):
        return env.next_infected_probabilities(a_) 
      def qfn_gnn(a_):
        X_ = env.data_block_at_action(-1, a_, raw=True)
        return predictor(X_)
      def qfn_l(a_):
        X_ = env.data_block_at_action(-1, a_, neighbor_order=2)
        return clf.predict_proba(X_)[:, 1]

      #GNN treatment
      argmaxer1 = argmaxer_factory('quad_approx')
      treat_gnn = argmaxer1(qfn_gnn, evaluation_budget, treatment_budget, env)
      #Linear treatment
      argmaxer2 = argmaxer_factory('sweep')
      treat_l = argmaxer2(qfn_l, evaluation_budget, treatment_budget, env)

      Q1_gnn = oracle_qfn(treat_gnn).sum()
      Q1_l = oracle_qfn(treat_l).sum()
      print(f'Q1_gnn: {Q1_gnn} Q1_l: {Q1_l}')

      #Test sweep and quad_approx
      treat_oracle_sweep = argmaxer1(oracle_qfn, evaluation_budget, treatment_budget, env)
      Q1_sweep = oracle_qfn(treat_oracle_sweep).sum()
      treat_oracle_quad_approx = argmaxer2(oracle_qfn, evaluation_budget, treatment_budget, env)
      Q1_quad_approx = oracle_qfn(treat_oracle_quad_approx).sum()
      print(f'Q1_sweep: {Q1_sweep} Q1_quad_approx: {Q1_quad_approx}')

      self.env.step(self.random_policy(**self.policy_arguments)[0])
    episode_results['acc_gnn'] = Acc_gnn
    episode_results['acc_hc'] = Acc_l
    
    return {replicate: episode_results}

class Simulator_treat1(object):
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

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return

  def run(self):
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 6))
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
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    Acc_gnn = []
    Acc_l = []

    for t in range(self.time_horizon-5):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None 
      
      print(self.env.Y[-1, :].mean())
      
      def oracle_qfn(a_):
        return env.next_infected_probabilities(a_) 

      argmaxer1 = argmaxer_factory('sweep')
      argmaxer2 = argmaxer_factory('quad_approx')
      argmaxer3 = argmaxer_factory('searching')
      #Test sweep and quad_approx
      treat_oracle_sweep = argmaxer1(oracle_qfn, evaluation_budget, treatment_budget, env)
      Q1_sweep = oracle_qfn(treat_oracle_sweep).sum()
      treat_oracle_quad_approx = argmaxer2(oracle_qfn, evaluation_budget, treatment_budget, env)
      Q1_quad_approx = oracle_qfn(treat_oracle_quad_approx).sum()
      treat_oracle_searching = argmaxer3(oracle_qfn, evaluation_budget, treatment_budget, env)
      Q1_searching = oracle_qfn(treat_oracle_searching).sum()

      print(f'Q1_sweep: {Q1_sweep} Q1_quad_approx: {Q1_quad_approx} Q1_searching : {Q1_searching}')

      self.env.step(self.random_policy(**self.policy_arguments)[0])
    episode_results['acc_gnn'] = Acc_gnn
    episode_results['acc_hc'] = Acc_l
    
    return {replicate: episode_results}
  
class Simulator_treat2(object):
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

  def save_results(self, results_dict):
    save_dict = {'settings': self.settings,
                 'results': results_dict}
    prefix = os.path.join(pkg_dir, 'analysis', 'results', self.basename)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(save_dict, outfile)
    return

  def run(self):
    np.random.seed(self.seed)
    num_processes = np.min((self.number_of_replicates, 6))
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
    episode_results = {'acc-gnn': None, 'acc-hc': None, 'score': None}
    self.env.reset()

    #Initial steps
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    self.env.step(self.random_policy(**self.policy_arguments)[0])
    Acc_gnn = []
    Acc_l = []

    for t in range(self.time_horizon-5):
      print(t)
      #########START FROM HERE#########
      classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
      self.policy_arguments['classifier'], self.policy_arguments['regressor'], self.policy_arguments['env'], \
      self.policy_arguments['evaluation_budget'], self.policy_arguments['treatment_budget'], self.policy_arguments['argmaxer'], \
      self.policy_arguments['bootstrap'], self.policy_arguments['gamma'], self.policy_arguments['raw_features']

      if bootstrap: 
        weights = np.random.exponential(size=len(env.X)*env.L)
      else:
        weights = None 
      
      #GNN
      _, predictor = learn_ggcn(env.X_raw[3:], env.y[3:], env.adjacency_list, neighbor_order=1)
      
      #Logistic regression
      #clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      clf = LogisticRegression()
      y = np.hstack(env.y[3:])
      X = np.vstack(env.X[3:])
      clf.fit(X, y)

      #testing
      acc_gnn = 0.
      acc_l = 0.
      for t in range(3):
         y_hat_gnn = (predictor(env.X_raw[t]) > 0.5)
         #y_hat_l = (clf.predict_proba(env.X[t], **predict_proba_kwargs) > 0.5)
         y_hat_l = (clf.predict_proba(env.X[t])[:, 1] > 0.5)
         acc_gnn += round((y_hat_gnn == env.y[t]).mean(), 3) / 3
         acc_l += round((y_hat_l == env.y[t]).mean(), 3) / 3
        
      Acc_gnn.append(acc_gnn)
      Acc_l.append(acc_l)

      print(f'acc_gnn: {acc_gnn} acc_l: {acc_l}')
      print(self.env.Y[-1, :].mean())

      def oracle_qfn(a_):
        return env.next_infected_probabilities(a_) 
      def qfn_gnn(a_):
        X_ = env.data_block_at_action(-1, a_, raw=True)
        return predictor(X_)
      def qfn_l(a_):
        X_ = env.data_block_at_action(-1, a_)
        return clf.predict_proba(X_)[:, 1]

      #compute absolute error and kl distance
      N_REP = 100 #从均匀分布中随机产生N_REP组治疗
      dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
      eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)] 

      linear_error = 0.
      gnn_error = 0.
      linear_distance = 0.
      gnn_distance = 0.
      for a_ in eval_actions:
        gnn_probs = qfn_gnn(a_)
        true_probs = oracle_qfn(a_)
        linear_probs = qfn_l(a_)
        linear_error += round(abs(np.array(linear_probs - true_probs)).sum(), 2) / N_REP
        gnn_error += round(abs(np.array(gnn_probs - true_probs)).sum(), 2) / N_REP
        linear_distance += round(kl(linear_probs, true_probs), 5) / N_REP
        gnn_distance += round(kl(gnn_probs, true_probs), 5) / N_REP
      print(f'gccn error: {gnn_error} linear error: {linear_error}')
      print(f'gccn distance: {gnn_distance} linear distance: {linear_distance}')

      #GNN treatment
      argmaxer1 = argmaxer_factory('quad_approx')
      treat_gnn_quad_qpprox = argmaxer1(qfn_gnn, evaluation_budget, treatment_budget, env)
      argmaxer2 = argmaxer_factory('sweep')
      treat_gnn_sweep = argmaxer2(qfn_gnn, evaluation_budget, treatment_budget, env)
      argmaxer3 = argmaxer_factory('searching')
      treat_gnn_searching = argmaxer3(qfn_gnn, evaluation_budget, treatment_budget, env)

      Q1_gnn_quad_approx = round(qfn_gnn(treat_gnn_quad_qpprox).sum(), 2)
      Q1_gnn_sweep = round(qfn_gnn(treat_gnn_sweep).sum(), 2)
      Q1_gnn_searching = round(qfn_gnn(treat_gnn_searching).sum(), 2)
      print(f'Q1_gnn_quad_approx: {Q1_gnn_quad_approx} Q1_gnn_sweep: {Q1_gnn_sweep} Q1_gnn_searching: {Q1_gnn_searching}')

      Q1_true_quad_approx = round(oracle_qfn(treat_gnn_quad_qpprox).sum(), 2)
      Q1_true_sweep = round(oracle_qfn(treat_gnn_sweep).sum(), 2)
      Q1_true_searching = round(oracle_qfn(treat_gnn_searching).sum(), 2)
      print(f'Q1_true_quad_approx: {Q1_true_quad_approx} Q1_true_sweep: {Q1_true_sweep} Q1_true_searching: {Q1_true_searching}')

      #GNN treatment
      treat_l_quad_qpprox = argmaxer1(qfn_gnn, evaluation_budget, treatment_budget, env)
      treat_l_sweep = argmaxer2(qfn_gnn, evaluation_budget, treatment_budget, env)
      treat_l_searching = argmaxer3(qfn_gnn, evaluation_budget, treatment_budget, env)

      Q1_l_quad_approx = round(qfn_l(treat_l_quad_qpprox).sum(), 2)
      Q1_l_sweep = round(qfn_l(treat_l_sweep).sum(), 2)
      Q1_l_searching = round(qfn_l(treat_l_searching).sum(), 2)
      print(f'Q1_l_quad_approx: {Q1_l_quad_approx} Q1_l_sweep: {Q1_l_sweep} Q1_l_searching: {Q1_l_searching}')

      Q1_true_quad_approx_l = round(oracle_qfn(treat_l_quad_qpprox).sum(), 2)
      Q1_true_sweep_l = round(oracle_qfn(treat_l_sweep).sum(), 2)
      Q1_true_searching_l = round(oracle_qfn(treat_l_searching).sum(), 2)
      print(f'Q1_true_quad_approx_l: {Q1_true_quad_approx_l} Q1_true_sweep_l: {Q1_true_sweep_l} Q1_true_searching_l: {Q1_true_searching_l}')

      treat_oracle_quad_approx = argmaxer1(oracle_qfn, evaluation_budget, treatment_budget, env)
      treat_oracle_sweep = argmaxer2(oracle_qfn, evaluation_budget, treatment_budget, env)
      treat_oracle_searching = argmaxer3(oracle_qfn, evaluation_budget, treatment_budget, env)

      Q1_oracle_quad_approx = round(oracle_qfn(treat_oracle_quad_approx).sum(), 2)
      Q1_oracle_sweep = round(oracle_qfn(treat_oracle_sweep).sum(), 2)
      Q1_oracle_searching = round(oracle_qfn(treat_oracle_searching).sum(), 2)
      print(f'Q1_oracle_quad_approx: {Q1_oracle_quad_approx} Q1_oracle_sweep: {Q1_oracle_sweep} Q1_oracle_searching: {Q1_oracle_searching}')

      Q1_gnn_oracle_treat = round(qfn_gnn(treat_oracle_quad_approx).sum(), 2)
      Q1_l_oracle_treat = round(qfn_l(treat_oracle_quad_approx).sum(), 2)
      print(f'Q1_gnn_oracle_treat: {Q1_gnn_oracle_treat} Q1_l_oracle_treat: {Q1_l_oracle_treat}')


      self.env.step(self.random_policy(**self.policy_arguments)[0])
    episode_results['acc_gnn'] = Acc_gnn
    episode_results['acc_hc'] = Acc_l
    
    return {replicate: episode_results}