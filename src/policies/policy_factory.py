'''
Description: 
version: 
Author: Zhishuai
Date: 2021-09-10 02:42:46
'''
import src.policies.reference_policies as ref
import src.policies.q_function_policies as roll
import src.policies.policy_search as ps



def policy_factory(policy_type):
  """
  :return: Corresponding policy function.
  """
  if policy_type == 'random':
    return ref.random
  elif policy_type == 'random_no_replace':
    return ref.random_no_replace
  elif policy_type == 'treat_first':
    return ref.treat_first
  elif policy_type == 'no_action':
    return ref.no_action
  elif policy_type == 'true_probs':
    return ref.true_probs
  elif policy_type == 'true_probs_myopic':
    return ref.true_probs_myopic
  elif policy_type == 'one_step':
    return roll.one_step_policy
  elif policy_type == 'two_step':
    return roll.two_step
  elif policy_type == 'three_step':
    return roll.three_step
  elif policy_type == 'one_step_gnn':
    return roll.one_step_gnn
  elif policy_type == 'two_step_gnn':
    return roll.two_step_gnn
  elif policy_type == 'three_step_gnn':
    return roll.three_step_gnn
  elif policy_type == 'treat_all':
    return ref.treat_all
  elif policy_type == 'policy_search':
    return ps.policy_search_policy
  elif policy_type == 'oracle_policy_search':
    return ps.oracle_policy_search_policy
  else:
    raise ValueError('Argument does not match any policy.')
