'''
Description: 
version: 
Author: Zhishuai
Date: 2021-09-10 02:42:46
'''
from .sweep.argmaxer_sweep import argmaxer_sweep
import numpy as np
import logging
import pdb
try:
  from scipy.special import comb
except ImportError:
  from scipy.misc import comb
from itertools import combinations
from .nonlinear.nonlinear import argmaxer_nonlinear

#随机生成treatment
def argmaxer_random(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  # Placeholder argmax function for debugging.
  if ixs is not None:
    L = len(ixs)
    treatment_budget = int(np.ceil((treatment_budget / env.L) * L))
  else:
    L = env.L
  dummy = np.append(np.ones(treatment_budget), np.zeros(L - treatment_budget))
  return np.random.permutation(dummy)

#从所有治疗组合中找出最优的
def argmaxer_global(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  HARD_EVALUATION_LIMIT = 1000
  #comb 组合数 C_{rho}^L
  assert(comb(env.L, treatment_budget) < HARD_EVALUATION_LIMIT,
         '(L choose treatment_budget) greater than HARD_EVALUATION_LIMIT.')
  all_ix_combos = combinations(range(env.L), treatment_budget)
  q_best = float('inf')
  a_best = None
  for ixs in all_ix_combos:
    a = np.zeros(env.L)
    for j in ixs:
      a[j] = 1
    q_sum = np.sum(q_fn(a))
    if q_sum < q_best: 
      q_best = q_sum
      a_best = a
  return a_best

def argmaxer_search(qfn, evaluation_budget, treatment_budget, env, ixs=None):
  N_REP=2000
  dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
  Qmin = 9999
  a_best = []
  for i in range(N_REP):
    eval_action = np.random.permutation(dummy_act)
    Q_temp = qfn(eval_action).sum()
    if Q_temp < Qmin:
      Qmin = Q_temp
      a_best = eval_action
  return a_best
    


def argmaxer_factory(choice):
  """
  :param choice: str in ['sweep', 'quad_approx']
  :return:
  """
  if choice == 'sweep':
    return argmaxer_sweep
  elif choice == 'searching':
    return argmaxer_search
  elif choice == 'quad_approx':
    try:
      from .quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
      return argmaxer_quad_approx
    except ImportError:
      return argmaxer_random
  elif choice == 'sequential_quad_approx':
    from .quad_approx.argmaxer_quad_approx import argmaxer_sequential_quad_approx
    return argmaxer_sequential_quad_approx
  elif choice == 'multiple_quad_approx':
    from .quad_approx.argmaxer_quad_approx import argmaxer_multiple_quad_approx
    return argmaxer_multiple_quad_approx
  elif choice == 'oracle_multiple_quad_approx':
    from .quad_approx.argmaxer_quad_approx import argmaxer_oracle_multiple_quad_approx
    return argmaxer_oracle_multiple_quad_approx
  elif choice == 'random':
    return argmaxer_random
  elif choice == 'global':
    logging.warning('Using global argmaxer; this may be especially slow.')
    return argmaxer_global
  elif choice == 'nonlinear':
    return argmaxer_nonlinear
  else:
    raise ValueError('Argument is not a valid argmaxer name.')
