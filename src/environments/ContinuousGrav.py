"""
Spread of disease in continuous space (pdf pg 13 of white nose paper);
locations are points in [0,1]^2.

This is still just the gravity model, though.
"""
import numpy as np
from scipy.special import expit
from src.environments.Gravity import Gravity
import src.environments.gravity_infection_probs as infection_probs


class ContinuousGrav(Gravity):
  # ToDo: These are placeholders!
  # The thetas are NOT numbered the same as in Nick's paper, because the Gravity class separates theta's associated
  # with x's and the other elements of theta.
  THETA_0 = 0.0
  THETA_1 = THETA_2 = THETA_3 = THETA_4 = 1.0
  THETA_x_l = np.ones(4)
  THETA_x_lprime = np.ones(4)

  # Covariate covariance kernel parameters
  # ToDo: Placeholders
  P = 4
  RHO = 10.0
  TAU = np.log(10.0)
  ETA = np.log(4.0)

  def __init__(self, L):
    adjacency_matrix = np.ones((L, L))  # Fully connected

    # Generate locations and pairwise distances
    self.location_coordinates = np.random.random(size=(L, L))
    distance_matrix = np.array([
      np.array([
        np.linalg.norm(x_l - x_lprime) for x_l in self.location_coordinates
      ])
      for x_lprime in self.location_coordinates])
    distance_matrix = (distance_matrix - np.mean(distance_matrix)) / np.std(distance_matrix)
    lambda_ = distance_matrix  # TODO: This is a placeholder!  Lambda should be something else (see paper).

    # Generate static covariates
    # From paper: For each location l, we generate four static covariates by using a mean 0 Gaussian process
    # with a multivariate separable isotropic covariance matrix that is exponential space and
    # autoregressive across the four covariates at each location.
    covariance_matrices = np.array([np.array([
      self.covariate_covariance(l, lprime) for l in range(L)
    ]) for lprime in range(L)])

    # Compute transmission_probs
    cov = self.covariate_covariance()
    x = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov)
    x = x.reshape((self.L, ContinuousGrav.P))
    z = np.floor(x[:, 0] - np.min(x[:, 0]))
    product_matrix = np.outer(z, z)
    initial_infections = np.random.binomial(1, 0.01, L)
    self.current_state = np.column_stack((x, z, initial_infections))
    Gravity.__init__(self, distance_matrix, product_matrix, adjacency_matrix, covariate_matrix,
                     np.array([ContinuousGrav.THETA_0, ContinuousGrav.THETA_1, ContinuousGrav.THETA_2,
                               ContinuousGrav.THETA_3, ContinuousGrav.THETA_4]),
                     ContinuousGrav.THETA_x_l, ContinuousGrav.THETA_x_lprime, lambda_, initial_infections)

  def covariate_covariance(self):
    covariance = np.zeros((self.L*ContinuousGrav.P, self.L*ContinuousGrav.P))
    for l in range(self.L):
      for lprime in range(l, self.L):
        x_l, x_lprime = self.location_coordinates[l, :], self.location_coordinates[lprime, :]
        squared_dist = np.dot(x_l - x_lprime, x_l - x_lprime)
        for r in range(ContinuousGrav.P):
          for s in range(r, ContinuousGrav.P):
            cov_l_lprime_r_s = ContinuousGrav.RHO*np.exp(-ContinuousGrav.TAU*squared_dist -
                                                         ContinuousGrav.ETA*np.abs(r-s))
            covariance[int(l*ContinuousGrav.P+r), int(lprime*ContinuousGrav.P+s)] = cov_l_lprime_r_s
            covariance[int(l*ContinuousGrav.P+s), int(lprime*ContinuousGrav.P+r)] = cov_l_lprime_r_s
            covariance[int(lprime*ContinuousGrav.P+r), int(l*ContinuousGrav.P+s)] = cov_l_lprime_r_s
            covariance[int(lprime*ContinuousGrav.P+s), int(l*ContinuousGrav.P+r)] = cov_l_lprime_r_s
    return covariance

  def reset(self):
    super(ContinuousGrav, self).reset()

  def feature_function(self, raw_data_block):
    pass

  def feature_function_at_action(self, old_data_block, old_action, action):
    pass

  def feature_function_at_location(self, l, raw_data_block):
    pass


