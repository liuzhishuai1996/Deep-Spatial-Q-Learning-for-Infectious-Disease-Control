# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:02:06 2018

@author: Jesse
"""

import pdb
import numpy as np 
"""
Parameter descriptions 

evaluation budget: number of treatment combinations to evaluate
treatment_budget: size of treated state subset
"""


def q(a, data_block_ix, env, predictive_model, raw=False, condition_on_infection_status=False, neighbor_order=1):
  data_block = env.data_block_at_action(data_block_ix, a, raw=raw, neighbor_order=neighbor_order)

  # if not raw data, we may need to provide indices of infected and not-infected locations
  if condition_on_infection_status:
    infections_at_ix = env.Y[data_block_ix, :]
    infected_locations = np.where(infections_at_ix == 1)[0]
    not_infected_locations = np.where(infections_at_ix == 0)[0]
    predictive_model_kwargs = {'infected_locations': infected_locations, 'not_infected_locations': not_infected_locations}
  else:
    predictive_model_kwargs = {}

  q_hat = predictive_model(data_block, **predictive_model_kwargs)
  return q_hat


def q_at_state_and_infection(a, env, predictive_model, S, Y, raw=False, neighbor_order=1,
                             condition_on_infection_status=False):
  """
  Evaluate q function at provided state and action, rather than the one at env.X_raw[data_block_ix].
  """
  data_block = np.column_stack((S, a, Y))
  if not raw:
    data_block = env.psi(data_block, neighbor_order=neighbor_order)
    if condition_on_infection_status:
      infected_locations = np.where(Y == 1)[0]
      not_infected_locations = np.where(Y == 0)[0]
      predictive_model_kwargs = {'infected_locations': infected_locations, 'not_infected_locations': not_infected_locations}
    else:
      predictive_model_kwargs = {}
  else:
    predictive_model_kwargs = {}
  return predictive_model(data_block, **predictive_model_kwargs)


def q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model, argmaxer,
                     list_of_infections_and_states=None, return_blocks_at_argmax=False, raw=False, neighbor_order=1,
                     condition_on_infection_status=False):
  """
  Take q_max for all data blocks in env history (need this for Q-learning/fqi).

  :param list_of_infections_and_states: list of tuples [S_t, Y_t] at which to evaluate q_max, instead of env.X
  """
  q_max_list = []
  argmax_list = []
  if return_blocks_at_argmax:
    block_at_argmax_list = []
  else:
    block_at_argmax_list = None
  for t in range(env.T):

    def q_fn(a):
      if list_of_infections_and_states is None:
        return q(a, t, env, predictive_model, raw=raw, neighbor_order=neighbor_order,
                 condition_on_infection_status=condition_on_infection_status)
      else:
        S_t, Y_t = list_of_infections_and_states[t]
        return q_at_state_and_infection(a, env, predictive_model, S_t, Y_t, raw=raw, neighbor_order=neighbor_order,
                                        condition_on_infection_status=condition_on_infection_status)

    argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env)
    if return_blocks_at_argmax:
      block_at_argmax = env.data_block_at_action(t, argmax, raw=raw, neighbor_order=neighbor_order)
      block_at_argmax_list.append(block_at_argmax)
    argmax_list.append(argmax)
    q_max_list.append(q_fn(argmax))
  return np.array(q_max_list), argmax_list, block_at_argmax_list
