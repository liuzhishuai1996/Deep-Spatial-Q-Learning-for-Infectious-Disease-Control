import pdb
import numpy as np
import copy


def perturb_action(action, number_to_perturb):
  """
  Randomly shuffles some treatments in action (for stochastic optimization of
  Q_fn).
  :param action:
  :param number_to_perturb:
  :return:
  """
  one_ixs = np.where(action == 1)[0]
  zero_ixs = np.where(action == 0)[0]
  perturb_ixs = np.random.choice(number_to_perturb)
  action[one_ixs[perturb_ixs]] = 0
  action[zero_ixs[perturb_ixs]] = 1
  return action

def swap_action(q_fn, action):
  """
  Move lowest q_fn treatment to highest non-treated area.
  :param q_fn:
  :param action:
  :return:
  """
  q = q_fn(action)
  highest_untreated = np.intersect1d(np.argsort(q), np.where(action == 0)[0])[-1]
  lowest_treated = np.intersect1d(np.argsort(q), np.where(action == 1)[0])[0]
  if q[lowest_treated] < q[highest_untreated]:
    new_action = copy.copy(action)
    new_action[lowest_treated] = 0
    new_action[highest_untreated] = 1
    return new_action
  else:
    return None

def argmaxer_sweep(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  if ixs:
    L = len(ixs)
    treatment_budget = int(np.ceil((treatment_budget/env.L) * L))
  else:
    L = env.L
  a = np.zeros(L)
  while np.sum(a) < treatment_budget:
    q = -q_fn(a)
    a_new = np.intersect1d(np.argsort(q), np.where(a == 0)[0])[0]
    a[a_new] = 1
  best_q = np.sum(q_fn(a))
  evaluation_counter = treatment_budget
  while evaluation_counter < evaluation_budget:
    a_new = swap_action(q_fn, a)
    evaluation_counter += 1
    if a_new is not None:
      q_new = np.sum(q_fn(a_new))
      if q_new < best_q:
        a = a_new
        best_q = q_new
    else:
      a_new = perturb_action(a, 1)
      q_new = np.sum(q_fn(a_new))
      evaluation_counter += 1
      if np.sum(q_fn(a_new)) < best_q:
        best_q = q_new
        a = a_new
  return a