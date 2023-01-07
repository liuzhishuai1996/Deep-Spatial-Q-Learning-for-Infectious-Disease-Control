import math
import numpy as np
from itertools import combinations
from src.utils.misc import onehot


def nCk(n, r):
  f = math.factorial
  return f(n) / f(r) / f(n - r)


def num_candidate_states(evaluation_budget, treatment_budget):
  """
  :return num_candidates: number of actions we can afford to evaluate under these budget
  """
  num_candidates = treatment_budget
  while nCk(num_candidates, treatment_budget) <= evaluation_budget:
    num_candidates += 1
  return num_candidates - 1


def all_candidate_actions(state_scores, evaluation_budget, treatment_budget):
  """
  :return candidate_actions: list of candidate actions according to state_scores
  """
  num_candidates = num_candidate_states(evaluation_budget, treatment_budget)
  sorted_indices = np.argsort(state_scores)
  candidate_indices = sorted_indices[:num_candidates]
  candidate_treatment_combinations = combinations(candidate_indices, treatment_budget)
  candidate_treatment_combinations = [list(combo) for combo in candidate_treatment_combinations]
  candidate_actions = np.zeros((0, len(state_scores)))
  for i in range(len(candidate_treatment_combinations)):
    a = np.zeros(len(state_scores))
    a[candidate_treatment_combinations[i]] = 1
    candidate_actions = np.vstack((candidate_actions, np.array(a)))
  return candidate_actions