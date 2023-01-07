import numpy as np
from src.environments.sis import SIS
from src.estimation.q_functions.q_functions import q
from src.utils.misc import random_argsort
from functools import partial


def simulate_from_SIS(env, eta, planning_depth, treatment_budget, n_rep=30):
  """
  For model-based RL in the sis generative model.

  :param env: sis object
  :param eta: length 7 array of disease probability parameters
  :param beta: length 2 array of state transition parameters
  :param planning_depth: how many steps ahead from current state to simulate
  :param argmaxer: dict of policy kwargs
  :param evaluation_budget:
  :param treatment_budget:
  :param n_rep: how many simulation replicates to run
  :return:
  """
  simulation_env = SIS(env.L, 0, None,
                       adjacency_matrix=env.adjacency_matrix,
                       initial_infections=env.current_infected,
                       initial_state=env.current_state,
                       add_neighbor_sums=env.add_neighbor_sums,
                       epsilon=env.epsilon,
                       contaminator=env.contaminator,
                       eta=eta)
  a = np.concatenate((np.zeros(simulation_env.L - treatment_budget), np.ones(treatment_budget)))
  for rep in range(n_rep):
    for t in range(planning_depth):
      # Use myopic estimated probs as rollout policy
      # print('rep {} t {}'.format(rep, t))
      a = np.zeros(L)
      probs = simulation_env.next_infected_probabilities(L)
      treat_ixs = random_argsort(-probs, treatment_budget)
      a[treat_ixs] = 1
      # a = np.random.permutation(a)
      # simulation_env.step(a)
    simulation_env.add_state(env.current_state)
    simulation_env.add_infections(env.current_infected)
  return simulation_env
