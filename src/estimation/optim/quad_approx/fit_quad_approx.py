"""
Fit quadratic approximation to Q function fixed at observed states (for approximating argmax Q as binary quadratic
program).
"""
import time
import pdb
import numpy as np
from src.estimation.optim.sweep.argmaxer_sweep import perturb_action
from sklearn.linear_model import LinearRegression, Ridge
from numba import njit, jit
import copy


# @njit
# @jit
def get_neighbor_ixn_features(a, neighbor_interactions):
  """
  Return treatments at l's neighbors and all neighbor treatment interactions, i.e.
  [a_j a_j*a_k] for j,k in neighbors(a_l) including a_l
  """
  neighbor_ixn_features = []
  for i in range(len(neighbor_interactions)):
    ixn = neighbor_interactions[i]
    neighbor_ixn_features.append(a[ixn[0]]*a[ixn[1]])
  return neighbor_ixn_features


def shuffle_random_bits(arr, num_to_shuffle):
  """
  Helper for doing sequential quad_approx.
  """
  L = len(arr)
  ixs = np.random.choice(L, size=num_to_shuffle)
  bits_to_shuffle = arr[ixs]
  np.random.shuffle(bits_to_shuffle)
  arr_copy = copy.copy(arr)
  arr_copy[ixs] = bits_to_shuffle
  return arr_copy


def sample_from_q(q, treatment_budget, evaluation_budget, L, initial_act):
  """
  Evaluate q function at evaluation_budget points in order to fit quadratic approximation.
  """
  num_to_shuffle = treatment_budget*2
  if initial_act is not None:
    acts_to_evaluate = [shuffle_random_bits(initial_act, num_to_shuffle) for _ in range(evaluation_budget)]
    acts_to_evaluate.append(initial_act)
  else:
    dummy_act = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
    acts_to_evaluate = [np.random.permutation(dummy_act) for e in range(evaluation_budget)]
  sample_qs = []
  for ix, act in enumerate(acts_to_evaluate):
    sample_qs.append(q(act))
  return np.array(sample_qs), acts_to_evaluate


def fit_quad_approx_at_location(sample_qs, sample_acts, l, l_ix, neighbor_interactions):
  reg = Ridge(alpha=5.0)
  X = np.array([get_neighbor_ixn_features(a, neighbor_interactions) for a in sample_acts])
  y = sample_qs[:, l_ix]
  reg.fit(X, y)
  return reg


def evaluate_quad_approx(reg_list, neighbor_interaction_lists, q, env_L, treatment_budget):
  sample_qs, sample_acts = sample_from_q(q, treatment_budget, 20, env_L, None)
  score = 0.
  for l, reg_l in enumerate(reg_list):
    neighbor_interactions = neighbor_interaction_lists[l]
    X_l = np.array([get_neighbor_ixn_features(a, neighbor_interactions) for a in sample_acts])
    y_l = sample_qs[:, l]
    score_l = reg_l.score(X_l, y_l)
    score += score_l / env_L
  return score


def fit_quad_approx(sample_qs, sample_acts, neighbor_interaction_lists, env_L, ixs, q, treatment_budget):
  quadratic_parameters = np.zeros((env_L, env_L))
  intercept = 0
  reg_list = []
  if ixs is None:
    ixs = range(env_L)
  for l_ix in ixs:
    l = ixs[l_ix]
    neighbor_interactions = neighbor_interaction_lists[l]
    if len(neighbor_interactions) > 0:
      reg = fit_quad_approx_at_location(sample_qs, sample_acts, l, l_ix, neighbor_interactions)
      intercept_l, beta_l = reg.intercept_, reg.coef_
      quadratic_parameters[neighbor_interactions[:, 0], neighbor_interactions[:, 1]] += beta_l
      intercept += intercept_l
      reg_list.append(reg)
  # score = evaluate_quad_approx(reg_list, neighbor_interaction_lists, q, env_L, treatment_budget)
  # print(f'score: {score}')
  return quadratic_parameters, intercept


def get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env, ixs, initial_act=None):
  if ixs is not None:
    L = len(ixs)
  else:
    L = env.L
  sample_qs, sample_acts = sample_from_q(q, treatment_budget, evaluation_budget, L, initial_act)
  quadratic_parameters, intercept = fit_quad_approx(sample_qs, sample_acts, env.neighbor_interaction_lists, env.L, ixs,
                                                    q, treatment_budget)
  return quadratic_parameters, intercept






