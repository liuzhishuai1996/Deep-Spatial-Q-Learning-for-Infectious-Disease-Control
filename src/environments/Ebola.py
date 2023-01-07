# -*- coding: utf-8 -*-
"""
Created on Wed May 16 23:18:41 2018

@author: Jesse
"""

import numpy as np
import copy
from scipy.special import expit
import src.environments.gravity_infection_probs as infection_probs
from src.environments.Gravity import Gravity
from src.environments.sis import SIS
from src.estimation.model_based.Gravity.estimate_ebola_parameters import log_lik_single
import src.utils.gradient as gradient
import pickle as pkl
import os
import pdb


class Ebola(Gravity):
  # Load network information
  this_file_pathname = os.path.dirname(os.path.abspath(__file__))
  ebola_network_data_fpath = os.path.join(this_file_pathname, 'ebola-network-data', 'ebola_network_data.p')
  network_info = pkl.load(open(ebola_network_data_fpath, 'rb'))
  # ADJACENCY_MATRIX = network_info['adjacency_matrix']

  # DISTANCE_MATRIX  = network_info['haversine_distance_matrix']
  DISTANCE_MATRIX = network_info['euclidean_distance_matrix']
  SUSCEPTIBILITY = network_info['pop_array'] / 10000
  L = len(SUSCEPTIBILITY)
  OUTBREAK_TIMES = network_info['outbreak_time_array']

  ADJACENCY_MATRIX = np.zeros((L, L))
  NEIGHBOR_DISTANCE_MATRIX = np.zeros((L, 3))
  for l in range(L):
    d_l_lprime = DISTANCE_MATRIX[l, :]
    s_l_lprime = SUSCEPTIBILITY[l] * SUSCEPTIBILITY
    sorted_ratios = np.argsort(d_l_lprime / s_l_lprime)
    j = 0
    for lprime in sorted_ratios[1:4]:
        ADJACENCY_MATRIX[l, lprime] = 1
        #NEIGHBOR_DISTANCE_MATRIX[l, j] = DISTANCE_MATRIX[l, ] + DISTANCE_MATRIX[j, l] ##### 这里我认为写错了 #####
        NEIGHBOR_DISTANCE_MATRIX[l, j] = DISTANCE_MATRIX[l, lprime] + DISTANCE_MATRIX[lprime, l]
        j += 1

  MAX_NUMBER_OF_NEIGHBORS = int(np.max(np.sum(ADJACENCY_MATRIX, axis=1)))
  # Fill matrix of susceptibility products
  PRODUCT_MATRIX = np.outer(SUSCEPTIBILITY, SUSCEPTIBILITY)

  # Get initial outbreaks
  OUTBREAK_TIMES[np.where(OUTBREAK_TIMES == -1)] = np.max(OUTBREAK_TIMES) + 1 # Make it easier to sort
  NUMBER_OF_INITIAL_OUTBREAKS = 25
  # NUMBER_OF_INITIAL_OUTBREAKS = int(np.floor(0.25 * L))
  OUTBREAK_INDICES = np.argsort(OUTBREAK_TIMES)[:NUMBER_OF_INITIAL_OUTBREAKS]
  INITIAL_INFECTIONS = np.zeros(L)
  INITIAL_INFECTIONS[OUTBREAK_INDICES] = 1 #选出最先爆发的25个地方作为初始感染

  # Params for logit of transmission probability
  ALPHA = 3.0
  # ALPHA = 2.0
  BETA = -5.0
  ETA_0 = SIS.ETA_2 * ALPHA
  ETA_1 = SIS.ETA_3 + np.log(ALPHA)
  # ETA_0 = SIS.ETA_2
  # ETA_1 = SIS.ETA_3
  ETA_2 = 0.0
  # ETA_3 = SIS.ETA_3
  # ETA_4 = SIS.ETA_4
  ETA_3 = ETA_4 = BETA

  # ALPHA = 1
  # ETA_0 = -8 * ALPHA
  # ETA_1 = np.log(156) + np.log(ALPHA)
  # ETA_2 = 5
  # ETA_3 = -8.0
  # ETA_4 = -8.0
  # ETA_0 = -7.2 * ALPHA
  # ETA_1 = -0.284 + np.log(ALPHA)
  # ETA_2 = -0.0
  # ETA_3 = -1.015 
  # ETA_4 = -1.015    
  ETA = np.array([ETA_0, ETA_1, ETA_2, ETA_3, ETA_4]) #不考虑recover
  # ETA = np.array([ETA_0, np.exp(ETA_1), np.exp(ETA_2), ETA_3, ETA_4])
  # Compute transmission probs

  def __init__(self, eta=None, construct_features_for_policy_search=False, learn_embedding=False):
    super(Ebola, self).__init__(Ebola.DISTANCE_MATRIX, Ebola.PRODUCT_MATRIX, Ebola.ADJACENCY_MATRIX,
                                Ebola.SUSCEPTIBILITY, Ebola.ETA, None, None, Ebola.ADJACENCY_MATRIX,
                                initial_infections=Ebola.INITIAL_INFECTIONS,
                                construct_features_for_policy_search=construct_features_for_policy_search,
                                learn_embedding=learn_embedding)

  # Neighbor features
  def second_order_encoding(self, l, raw_data_block):
    second_order_features = [0]*16
    for lprime in self.adjacency_list[l]:
      a_lprime, y_lprime = raw_data_block[lprime, 1], raw_data_block[lprime, 2]
      first_order_encoding = int(1*a_lprime + 2*y_lprime)
      for lprime_prime in self.adjacency_list[lprime]:
        a_lprime_prime, y_lprime_prime = raw_data_block[lprime_prime, 1], raw_data_block[lprime_prime, 2]
        second_order_encoding = first_order_encoding + int(4*a_lprime_prime + 8*y_lprime_prime)
        second_order_features[second_order_encoding] += 1
    return second_order_features

  def feature_function_at_location(self, l, raw_data_block, neighbor_order=1):
    x_l = raw_data_block[l, :]
    neighbors = self.adjacency_list[l]
    num_neighbors = len(neighbors)

    # Concatenate features at each neighbor
    for i in range(self.MAX_NUMBER_OF_NEIGHBORS):
      if i >= num_neighbors:  # zero padding
        x_l = np.concatenate((x_l, np.zeros(4)))
      else:
        l_prime = neighbors[i]
        x_lprime = raw_data_block[l_prime, :]
        x_l = np.concatenate((x_l, x_lprime, [self.NEIGHBOR_DISTANCE_MATRIX[l, i]]))

    # If second-order, concatenate counts of infection and treatment status pairs for first- and second-order neighbors
    # (similar to SIS).
    if neighbor_order == 2:
      second_order_features = self.second_order_encoding(l, raw_data_block)
      x_l = np.concatenate((x_l, second_order_features))
    return x_l

  def feature_function(self, raw_data_block, neighbor_order=1):
    # (s, a, y) for location and (s, a, y, d) for each of its neighbors
    number_of_features = int(3 + 4*self.MAX_NUMBER_OF_NEIGHBORS)
    if neighbor_order == 2:
      number_of_features += 16
    X = np.zeros((0, number_of_features))
    for l in range(self.L):
      x_l = self.feature_function_at_location(l, raw_data_block, neighbor_order=neighbor_order)
      X = np.vstack((X, x_l))
    return X

  def feature_function_at_action(self, old_data_block, old_action, action, neighbor_order=1):
    new_data_block = copy.copy(old_data_block)
    locations_with_changed_actions = set(np.where(old_action != action)[0])

    for l in range(self.L):
      # Check if action at l changed
      if l in locations_with_changed_actions:
        new_data_block[l, 1] = action[l]

      # Check if actions at l neighbors have changed
      for i in range(len(self.adjacency_list[l])):
        l_prime = self.adjacency_list[l][i]
        if l_prime in locations_with_changed_actions:
          new_data_block[l, 3 + i*4 + 1] = action[l_prime]

    # If neighbor_order=2, have to account for changes in second-order neighbors, too.
    if neighbor_order == 2:
      new_raw_data_block = new_data_block[:, :4]
      new_second_order_features = np.zeros((0, 16))
      for l in range(self.L):
        new_second_order_features_l = self.second_order_encoding(l, new_raw_data_block)
        new_second_order_features = np.vstack((new_second_order_features, new_second_order_features_l))
      new_data_block = np.column_stack((new_data_block, new_second_order_features))
    return new_data_block

  def mb_covariance(self, mb_params):
    """
    Compute covariance of mb estimator.

    :param mb_params:
    :return:
    """
    dim = len(mb_params)
    grad_outer= np.zeros((dim, dim))
    hess = np.zeros((dim, dim))

    for t in range(self.T):
      data_block, raw_data_block = self.X[t], self.X_raw[t]
      a, y = raw_data_block[:, 1], raw_data_block[:, 2]
      y_next = self.y[t]
      for l in range(self.L):
        x_raw, x = raw_data_block[l, :], data_block[l, :]

        # gradient
        if raw_data_block[l, 2]:
          # No recovery
          pass
        else:
          def mb_log_lik_at_x(mb_params_):
            lik = mb_log_lik_single(a, y, y_next, l, self.L, mb_params_, self.ADJACENCY_MATRIX, self.DISTANCE_MATRIX,
                                    self.PRODUCT_MATRIX)
            return lik

          mb_grad = gradient.central_diff_grad(mb_log_lik_at_x, mb_params)
          mb_hess = gradient.central_diff_hess(mb_log_lik_at_x, mb_params)

          grad_outer_lt = np.outer(mb_grad, mb_grad)
          grad_outer += grad_outer_lt
          hess += mb_hess

    hess_inv = np.linalg.inv(hess + 0.1 * np.eye(dim))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(self.L * self.T)

    # Check if array is finite
    try:
      np.asarray_chkfinite(cov)
    except ValueError:
      print('hess inv: {}\ngrad outer: {}'.format(hess_inv, grad_outer))
      # Some ad-hoc bullshit
      hess_inv = np.linalg.inv(np.diag(np.diag(hess + 0.1 * np.eye(dim))))
      cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(self.L * self.T)
    return cov

    
def mb_log_lik_single(a, y, y_next_, l, L, eta, adjacency_matrix, distance_matrix, product_matrix):
  eta0, exp_eta1, exp_eta2, eta3, eta4 = \
    eta[0], np.exp(eta[1]), np.exp(eta[2]), eta[3], eta[4]
  return log_lik_single(a, np.array(y), np.array(y_next_), l, L, eta0, exp_eta1, exp_eta2, eta3, eta4, adjacency_matrix,
                        distance_matrix, product_matrix)

