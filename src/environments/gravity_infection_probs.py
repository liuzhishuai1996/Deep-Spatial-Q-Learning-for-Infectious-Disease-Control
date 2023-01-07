import numpy as np
from scipy.special import expit
from numba import njit


@njit
def expit2(x):
  """
  To use with njit.

  :param x:
  :return:
  """
  exp_ = np.exp(-x)
  return 1.0 - 1.0 / (1 + exp_)


def ebola_infection_probs(a, y, eta, L, adjacency_lists, **kwargs):
  distance_matrix, adjacency_matrix, product_matrix, x = \
    kwargs['distance_matrix'], kwargs['adjacency_matrix'], kwargs['product_matrix'], kwargs['x']
  if x is not None:
    eta_x_l, eta_x_lprime = kwargs['eta_x_l'], kwargs['eta_x_lprime']
  else:
    eta_x_l, eta_x_lprime = None, None
  return np.array([infection_prob_at_location(a, l, eta, y, adjacency_lists, distance_matrix,
                                              product_matrix, x, eta_x_l, eta_x_lprime) for l in range(L)])


@njit
def get_all_gravity_transmission_probs_without_covariate_njit(a, eta, L, distance_matrix, adjacency_matrix,
                                                              product_matrix):
  transmission_probs_matrix = np.zeros((L, L))
  for l in range(L):
    for lprime in range(l, L):
      if adjacency_matrix[l, lprime]:
        d_l_lprime = distance_matrix[l, lprime]
        log_grav_term = np.log(d_l_lprime) - np.exp(eta[2]) * np.log(product_matrix[l, lprime])
        baseline_logit = eta[0] - np.exp(eta[1] + log_grav_term)
        transmission_prob_l_lprime = expit2(baseline_logit + a[l] * eta[3] + a[lprime] * eta[4])
        transmission_prob_lprime_l = expit2(baseline_logit + a[lprime] * eta[3] + a[l] * eta[4])
        transmission_probs_matrix[l, lprime] = transmission_prob_l_lprime
        transmission_probs_matrix[lprime, l] = transmission_prob_lprime_l
  return transmission_probs_matrix


@njit
def get_all_gravity_transmission_probs_with_covariate_njit(a, eta, L, distance_matrix, adjacency_matrix, product_matrix,
                                                          x, eta_x_l, eta_x_lprime):
  """
  Equation 2 of white nose paper,  eta in our notation is different than theta in their notation...


  :param a:
  :param x:
  :param l:
  :param l_prime:
  :param eta:
  :param L:
  :param kwargs:
  :return:
  """
  transmission_probs_matrix = np.zeros((L, L))

  for l in range(L):
    x_l = x[l, :]
    for lprime in range(l + 1, L):
      if adjacency_matrix[l, lprime]:
        d_l_lprime = distance_matrix[l, lprime]
        product_l_lprime = product_matrix[l, lprime]
        x_lprime = x[lprime, :]
        logit_l_lprime = eta[0] + theta[1]*d_l_lprime / np.power(product_l_lprime, eta[2]) - a[l]*eta[3] - \
                         a[lprime]*eta[4] + np.dot(eta_x_l, x_l) + np.dot(eta_x_lprime, x_lprime)
        logit_lprime_l = eta[0] + theta[1]*d_l_lprime / np.power(product_l_lprime, eta[2]) - a[lprime]*eta[3] - \
                         a[l]*eta[4] + np.dot(eta_x_l, x_lprime) + np.dot(eta_x_lprime, x_l)
        transmission_probs_matrix[l, lprime] = expit2(logit_l_lprime)
        transmission_probs_matrix[l, lprime] = expit2(logit_lprime_l)
  return transmission_probs_matrix


def get_all_gravity_transmission_probs(a, eta, L, **kwargs):
  distance_matrix, adjacency_matrix, product_matrix, x = \
    kwargs['distance_matrix'], kwargs['adjacency_matrix'], kwargs['product_matrix'], kwargs['x']
  if x is not None:
    eta_x_l, eta_x_lprime = kwargs['eta_x_l'], kwargs['eta_x_lprime']
  else:
    eta_x_l, eta_x_lprime = None, None

  if x is None:
    return get_all_gravity_transmission_probs_without_covariate_njit(a, eta, L, distance_matrix, adjacency_matrix,
                                                                     product_matrix)
  else:
    return get_all_gravity_transmission_probs_with_covariate_njit(a, eta, L, distance_matrix, adjacency_matrix,
                                                                  product_matrix, x, eta_x_l, eta_x_lprime)


def gravity_transmission_probs(a, l, lprime, eta, covariate_matrix, distance_matrix, product_matrix, x,
                               eta_x_l, eta_x_lprime):
  """

  :param l:
  :param lprime:
  :param a:
  :param eta:
  :param distance_matrix:
  :param product_matrix:
  :param x: Array of covariates, or None
  :param eta_x_l: coefficients for x_l, or None
  :param eta_x_lprime: coefficients for x_lprime, or None
  :return:
  """

  l, lprime = int(l), int(lprime)
  d_l_lprime = distance_matrix[l, lprime]
  product_l_lprime = product_matrix[l, lprime]
  logit = eta[0] + eta[1]*d_l_lprime / np.power(product_l_lprime, eta[2]) - a[l]*eta[3] - a[lprime]*eta[4]
  if x is not None:
    x_l, x_lprime = x[l, :], x[lprime, :]
    logit += np.dot(eta_x_l, x_l) + np.dot(eta_x_lprime, x_lprime)
  return expit2(logit)


def infection_prob_at_location(a, l, eta, current_infected, adjacency_list, distance_matrix, product_matrix, x,
                               eta_x_l, eta_x_lprime):
  if current_infected[l]:
    return 1
  else:
    not_transmitted_prob = np.product([1 - gravity_transmission_probs(a, l, l_prime, eta, None, distance_matrix,
                                                                      product_matrix, x, eta_x_l, eta_x_lprime)
                                       for l_prime in adjacency_list[l]])
    inf_prob = 1 - not_transmitted_prob
    return inf_prob
