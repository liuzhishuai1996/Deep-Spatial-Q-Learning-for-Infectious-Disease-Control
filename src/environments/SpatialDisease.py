# -*- coding: utf-8 -*-
"""
Created on Thu May 17 20:58:36 2018

@author: Jesse
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import networkx as nx
import pdb
from ..estimation.q_functions.embedding import learn_ggcn

# Create ABC base class compatible with Python 2.7 and 3.x
ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class SpatialDisease(ABC):
  INITIAL_INFECT_PROP = 0.1

  def __init__(self, adjacency_matrix, initial_infections=None, construct_features_for_policy_search=False,
               regenerate_network=False, compute_pairwise_distances=False, learn_embedding=False):
    """
    :param adjacency_matrix: 2d binary array corresponding to network for gen model
    :param initial_infections: L-length binary array of initial infections, or None
    """

    self.initial_infections = initial_infections
    self.construct_features_for_policy_search = construct_features_for_policy_search
    self.compute_pairwise_distances = compute_pairwise_distances
    self.embedder = None
    self.learn_embedding = learn_embedding
    # Generative model parameters
    self.L = adjacency_matrix.shape[0]
    self.max_number_of_neighbors = int(np.max(np.sum(adjacency_matrix, axis=1)))

    # Adjacency info
    self.construct_network(adjacency_matrix, construct_features_for_policy_search, compute_pairwise_distances)

    # Observation history
    if self.initial_infections is None:
      number_initial_infections = int(self.INITIAL_INFECT_PROP * self.L)
      initial_infect_indices = np.random.choice(self.L, number_initial_infections, replace=False)
      self.Y = np.zeros((1, self.L))
      self.Y[0, initial_infect_indices] = 1
    else:
      self.Y = np.array([self.initial_infections])
    self.A = np.zeros((0, self.L))
    self.X_raw = []  # Will hold blocks [S_t, A_t, Y_t] at each time t
    self.X = []
    self.X_2 = []
    self.y = []  # Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = []

    # Current network status
    self.current_infected = self.Y[-1, :]
    self.T = 0

  def construct_network(self, adjacency_matrix, construct_features_for_policy_search, compute_pairwise_distances):
    self.L = adjacency_matrix.shape[0]

    # Adjacency info
    self.adjacency_matrix = adjacency_matrix
    self.adjacency_list = np.array([np.array([l_prime for l_prime in range(self.L)
                                              if self.adjacency_matrix[l, l_prime] == 1])
                                    for l in range(self.L)], dtype = object) ##### 这里我加上了dtype = object 为了不报警告 #####

    network_as_nx_object = nx.from_numpy_matrix(self.adjacency_matrix)
    if construct_features_for_policy_search or compute_pairwise_distances:
      pairwise_distance_dictionary = dict(nx.all_pairs_shortest_path_length(network_as_nx_object))
      self.pairwise_distances = np.zeros((self.L, self.L))  # Entries are omega's in Nick's WNS paper
      for source_index, targets in pairwise_distance_dictionary.items():
        for target_index, length in targets.items():
          self.pairwise_distances[source_index, target_index] = length
          self.pairwise_distances[target_index, source_index] = length

      data_depth_dictionary = nx.algorithms.centrality.subgraph_centrality(network_as_nx_object)
      self.data_depth = np.zeros(self.L)
      for node_ix, subgraph_centrality in data_depth_dictionary.items():
        self.data_depth[node_ix] = subgraph_centrality

    self.num_neighbors = [len(neighbors) for neighbors in self.adjacency_list]
    self.num_neighbors_rep = [self.num_neighbors]
    self.neighbor_interaction_lists = [
      np.array([[i, j] for i in self.adjacency_list[l] for j in self.adjacency_list[l]])
      for l in range(self.L)]

  def reset(self):
    """
    Reset state and observation histories.
    """
    # Observation history
    if self.initial_infections is None:
      number_initial_infections = int(self.INITIAL_INFECT_PROP * self.L)
      initial_infect_indices = np.random.choice(self.L, number_initial_infections, replace=False)
      self.Y = np.zeros((1, self.L))
      self.Y[0, initial_infect_indices] = 1
    else:
      self.Y = np.array([self.initial_infections])
    self.A = np.zeros((0, self.L))
    self.X_raw = []
    self.X = []
    self.X_2 = []
    self.y = [] # Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = []

    # Current network status
    self.current_infected = self.Y[-1, :]
    self.T = 0

  @abstractmethod
  def update_obs_history(self, a):
    pass

  @abstractmethod
  def next_state(self):
    pass

  @abstractmethod
  def next_infections(self, a, eta=None):
    pass

  def fit_embedding(self):
    self.embedder, self.predictor = learn_ggcn(self.X_raw, self.y, self.adjacency_list)


  def step(self, a, eta=None):
    """
    Move model forward according to action a. 
    :param a: self.L-length array of binary actions at each state
    :param eta:
    """
    self.A = np.vstack((self.A, a)) #每一期的治疗按行摞起来
    self.next_infections(a, eta) #更新感染状态（结局）
    self.next_state() #更新状态
    self.update_obs_history(a) #保存新产生的信息
    self.T += 1

  @abstractmethod
  def data_block_at_action(self, data_block, action):
    pass
