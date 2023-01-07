
"""
Spread of disease in continuous space (pdf pg 13 of white nose paper);
locations are points in [0,1]^2.

This is still just the gravity model, though.
"""
import numpy as np
from abc import abstractmethod
import pdb
from scipy.special import expit
from src.environments.SpatialDisease import SpatialDisease
import src.environments.gravity_infection_probs as infection_probs
import copy


class Gravity(SpatialDisease):
  """
  Class for gravity disease models.  Gravity and the generic Continuous class will inherit from this.

  Transmission probabilities in gravity model are of the form

  logit(p_l_lprime) = \theta_0 + <\theta_1 | x_l> + <\theta_2 | x_lprime> - \theta_3*a_l - \theta_4*a_l
    - \theta_5 * d_l_lprime / (product_l_lprime)**\theta_6

  where (x_l, x_lprime) are optional covariates (Gravity doesn't have any).
  """
  def __init__(self, distance_matrix, product_matrix, adjacency_matrix, covariate_matrix, theta,
               theta_x_l, theta_x_lprime, lambda_, initial_infections=None,
               construct_features_for_policy_search=False, learn_embedding=False):
    """

    :param distance_matrix:
    :param product_matrix:
    :param adjacency_matrix:
    :param covariate_matrix: L x (covariate dimension) array, or None (as in Gravity)
    :param lambda_: array of weights used in policy search
    """
    SpatialDisease.__init__(self, adjacency_matrix, initial_infections=initial_infections,
                            construct_features_for_policy_search=construct_features_for_policy_search,
                            learn_embedding=learn_embedding)
    self.current_state = None  # TODO

    self.distance_matrix = distance_matrix
    self.product_matrix = product_matrix
    self.covariate_matrix = covariate_matrix
    self.include_covariates = (theta_x_l is not None)
    self.theta = theta
    self.theta_x_l = theta_x_l
    self.theta_x_lprime = theta_x_lprime
    self.L = distance_matrix.shape[0]
    self.precompute_transmission_probs()
    self.lambda_ = lambda_

    # Initial steps
    self.step(np.zeros(self.L))
    self.step(np.zeros(self.L))

  def precompute_transmission_probs(self):
    """
    Pre-compute transmission probabilities, since they depend only on actions and fixed covariates.

    :return:
    """
    self.transmission_probs = np.zeros((self.L, self.L, 2, 2))
    for l in range(self.L):
      for lprime in range(self.L):
        if self.adjacency_matrix[l, lprime] + self.adjacency_matrix[lprime, l] > 0:
          d_l_lprime = self.distance_matrix[l, lprime]
          product_l_lprime = self.product_matrix[l, lprime]
          log_grav_term = np.log(d_l_lprime) - np.exp(self.theta[2])*np.log(product_l_lprime)
          baseline_logit = self.theta[0] - np.exp(self.theta[1] + log_grav_term)
          if self.include_covariates:
            x_l = self.covariate_matrix[l, :]
            x_lprime = self.covariate_matrix[lprime, :]
            baseline_logit += np.dot(self.theta_x_l, x_l) + np.dot(self.theta_x_lprime, x_lprime)
          self.transmission_probs[l, lprime, 0, 0] = expit(baseline_logit)
          self.transmission_probs[l, lprime, 1, 0] = expit(baseline_logit + self.theta[3])
          self.transmission_probs[l, lprime, 0, 1] = expit(baseline_logit + self.theta[4])
          self.transmission_probs[l, lprime, 1, 1] = expit(baseline_logit + self.theta[3] + self.theta[4])

  def transmission_prob(self, a, l, lprime, eta, x, eta_x_l, eta_x_lprime):
    if self.current_infected[lprime]:
      if eta is None:
        transmission_prob = self.transmission_probs[l, lprime, int(a[l]), int(a[lprime])]
      else:
        transmission_prob = infection_probs.gravity_transmission_probs(a, l, lprime, eta, self.covariate_matrix,
                                                                       self.distance_matrix, self.product_matrix,
                                                                       x, eta_x_l, eta_x_lprime)
    else:
      transmission_prob = 0.0
    return transmission_prob

  def infection_probability(self, a, y, x):
    return np.array([
      self.infection_prob_at_location(a, y, l, None, x, None, None) for l in range(self.L)
    ])

  def infection_prob_at_location(self, a, y, l, eta, x, eta_x_l, eta_x_lprime):
    if y[l] == 1:
      return 1.0
    else:
      not_transmitted_prob = np.product([1 - self.transmission_prob(a, l, lprime, eta, x, eta_x_l, eta_x_lprime)
                                         for lprime in self.adjacency_list[l]])
      inf_prob = 1 - not_transmitted_prob
      return inf_prob

  def next_infected_probabilities(self, a, x=None, eta=None, eta_x_l=None, eta_x_lprime=None):
    return np.array([
      self.infection_prob_at_location(a, self.current_infected, l, eta, x, eta_x_l, eta_x_lprime) for l in range(self.L)
    ])

  @abstractmethod
  def feature_function(self, raw_data_block, neighbor_order=1):
    pass

  @abstractmethod
  def feature_function_at_action(self, old_data_block, old_action, action, neighbor_order=1):
    pass

  @abstractmethod
  def feature_function_at_location(self, l, raw_data_block, neighbor_order=1):
    pass

  def update_obs_history(self, a):
    super(Gravity, self).update_obs_history(a)
    raw_data_block = np.column_stack((self.covariate_matrix, a, self.Y[-2, :]))
    data_block = self.feature_function(raw_data_block)
    data_block_2 = self.feature_function(raw_data_block, neighbor_order=2)
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.X_2.append(data_block_2)
    self.y.append(self.current_infected)

  def binary_state_psi(self, s, y):
    return np.column_stack((s, y))

  def update_state(self, s):
    return self.covariate_matrix

  def next_state(self):
    super(Gravity, self).next_state()

  def next_infections(self, a, x=None, eta=None, eta_x_l=None, eta_x_lprime=None):
    next_infected_probabilities = self.next_infected_probabilities(a, x=x, eta=eta, eta_x_l=eta_x_l,
                                                                   eta_x_lprime=eta_x_lprime)
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.current_infected = next_infections

  def data_block_at_action(self, data_block_ix, action, neighbor_order=1, raw=False):
    super(Gravity, self).data_block_at_action(data_block_ix, action)
    if raw:
      new_data_block = copy.copy(self.X_raw[data_block_ix])
      new_data_block[:, 1] = action
    else:
      new_data_block = self.feature_function_at_action(self.X[data_block_ix], self.A[data_block_ix, :], action,
                                                       neighbor_order)
    return new_data_block

  def mb_covariance(self, mb_params):
    """
    Covariance of gravity model MLE.

    :param mb_params: MLE of \eta.
    :return:
    """
    pass

  def reset(self):
    super(Gravity, self).reset()
    self.step(np.zeros(self.L))
    self.step(np.zeros(self.L))

