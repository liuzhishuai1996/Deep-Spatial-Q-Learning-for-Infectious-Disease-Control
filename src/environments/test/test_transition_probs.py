import pdb
import numpy as np
from ..sis import SIS


def test_transition_probs_from_draft():
  # ToDo: split this into separate tests

  # Make sure we reproduce probs described on pg. 15 of the draft
  def generate_network(dummy):
    adjacency_matrix = np.zeros((4, 4))
    for i in range(1, 4):
      adjacency_matrix[0, i] = 1
      adjacency_matrix[i, 0] = 1
    return adjacency_matrix


  # (s, y, a) arrays to test at
  y_list = [np.zeros(4), np.array([0, 1, 1, 1]), np.array([0, 1, 1, 1]), np.array([0, 1, 1, 1]),
            np.ones(4), np.ones(4)]
  a_list = [np.zeros(4), np.zeros(4), np.array([1, 0, 0, 0]), np.array([0, 1, 1, 1]), np.zeros(4),
            np.ones(4)]
  true_probs_list = [0.01, 0.5, 0.5*0.75, 0.5*0.25, 1-0.25, (1-0.25)*0.5]

  # Test p_10
  env = SIS(lambda x: x, 4, 1, generate_network, initial_infections=y_list[0])
  phat = env.p_l0(a_list[0])
  assert np.isclose(phat[0], true_probs_list[0])

  # Test p_1
  for i in range(1, 4):
    env = SIS(lambda x: x, 4, 1, generate_network, initial_infections=y_list[i])
    phat = env.next_infected_probabilities(a_list[i])
    assert np.allclose(phat[0], true_probs_list[i], atol=0.005)

  # Test q_1
  for i in range(4, 6):
    env = SIS(lambda x: x, 4, 1, generate_network, initial_infections=y_list[i])
    qhat = env.next_infected_probabilities(a_list[i])
    print(qhat[0])
    assert np.isclose(qhat[0], true_probs_list[i])
