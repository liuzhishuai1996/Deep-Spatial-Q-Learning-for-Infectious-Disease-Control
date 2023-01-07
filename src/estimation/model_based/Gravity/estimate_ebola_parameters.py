import numpy as np
import pdb
import copy
from scipy.optimize import minimize
from numba import njit, jit
from functools import partial


@njit
def log_lik_single(a, y, y_next_, l, L, eta0, exp_eta1, exp_eta2, eta3, eta4, adjacency_matrix, distance_matrix,
                   product_matrix):
  # Log likelihood at location l at time t
  prod = 1.0
  a_l = a[l]
  for l_prime in range(L):
    # Transmissions only from infected neighbors
    if (adjacency_matrix[l, l_prime] == 1 or adjacency_matrix[l_prime, l] == 1) and y[l_prime]:
      # if y[l_prime]:
      d_l_lprime = distance_matrix[l, l_prime]
      s_l_lprime = product_matrix[l, l_prime]
      a_l_prime = a[l_prime]
      logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.power(s_l_lprime, exp_eta2) + \
                                eta3 * a_l + eta4 * a_l_prime
      one_minus_transmission_prob = 1.0 / (1.0 + np.exp(logit_transmission_prob))
      prod *= one_minus_transmission_prob

  if y_next_[l]:
    log_lik = np.log(1 - prod)
  else:
    log_lik = np.log(prod)

  return log_lik


@njit
def negative_log_likelihood(eta0, exp_eta1, exp_eta2, eta3, eta4, A, Y, y_next, distance_matrix, product_matrix,
                            adjacency_matrix, T, L, bootstrap_weights, indices_mask):
  log_lik = 0
  for t in range(T):
    a = A[t, :]
    y = Y[t, :]
    y_next_ = y_next[t, :]

    for l in range(L):
      if indices_mask[t, l] and not y[l]:  # Only model uninfected locations; infected locations never recover
        # log_lik_at_l = log_lik_single(a, y_next_, l, L, eta0, exp_eta1, exp_eta2, eta3, eta4, adjacency_matrix,
        #                               distance_matrix, product_matrix)
        # Log likelihood at location l at time t
        prod = 1.0
        a_l = a[l]
        for l_prime in range(L):
          # Transmissions only from infected neighbors
          if (adjacency_matrix[l, l_prime] == 1 or adjacency_matrix[l_prime, l] == 1) and y[l_prime]:
            # if y[l_prime]:
            d_l_lprime = distance_matrix[l, l_prime]
            s_l_lprime = product_matrix[l, l_prime]
            a_l_prime = a[l_prime]
            log_grav_term = np.log(d_l_lprime) - exp_eta2 * np.log(s_l_lprime)
            logit_transmission_prob = eta0 - exp_eta1 * np.exp(log_grav_term) + eta3*a_l + eta4*a_l_prime
            # logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.power(s_l_lprime, exp_eta2) + \
            #                           eta3 * a_l + eta4 * a_l_prime
            one_minus_transmission_prob = 1.0 / (1.0 + np.exp(logit_transmission_prob))
            prod *= one_minus_transmission_prob

        if y_next_[l]:
          log_lik_at_l = np.log(1 - prod)
        else:
          log_lik_at_l = np.log(prod)
        log_lik += log_lik_at_l * bootstrap_weights[t, l]

  return -log_lik


# ToDo: This turns out to be unnecessary...
def negative_log_likelihood_wrapper(eta, A, Y, y_next, distance_matrix, product_matrix, adjacency_matrix, T, L,
                                    bootstrap_weights, indices):
  eta0 = eta[0]
  eta1 = np.max((-10, np.min((10, eta[1]))))  # For stability
  eta2 = np.max((-10, np.min((10, eta[2]))))
  exp_eta1 = np.exp(eta1)
  exp_eta2 = np.exp(eta2)
  eta3 = eta[3]
  eta4 = eta[4]
  return negative_log_likelihood(eta0, exp_eta1, exp_eta2, eta3, eta4, A, Y, y_next, distance_matrix, product_matrix,
                                 adjacency_matrix, T, L, bootstrap_weights, indices)


def fit_ebola_transition_model(env, y_next=None, indices=None, bootstrap=False):
  """

  :param env:
  :param y_next: list of length-L binary vectors for infections after time t, or None;
                 used to get parametric bootstrap estimate of ebola model sampling dbn.
  :param indices:
  :return:
  """
  if y_next is None:
    y_next = np.array(env.y)
  if bootstrap:
    bootstrap_weights = np.random.exponential(scale=1.0, size=(env.T, env.L))
  else:
    bootstrap_weights = np.ones((env.T, env.L))

  if indices is None:
    indices_mask = np.ones((env.T, env.L))
  else:
    indices_mask = np.array([[np.float(l in indices[t]) for l in range(env.L)] for t in range(env.T)])

  objective = partial(negative_log_likelihood_wrapper, A=env.A, Y=env.Y, y_next=y_next,
                      distance_matrix=env.DISTANCE_MATRIX, product_matrix=env.PRODUCT_MATRIX,
                      adjacency_matrix=env.ADJACENCY_MATRIX, T=env.T, L=env.L, bootstrap_weights=bootstrap_weights,
                      indices=indices_mask)
  x0 = copy.copy(env.ETA)
  # x0[1] = np.log(x0[1] + 0.001)
  # x0[2] = np.log(x0[2] + 0.001)
  # x0 = np.random.normal(size=5)

  res = minimize(objective, x0=x0, method='L-BFGS-B')
  eta_hat = res.x
  negative_log_likelihood_ = objective(eta_hat)
  p = len(eta_hat)
  n = env.T * env.L
  aic = p + negative_log_likelihood_ + (p**2 + p) / np.max((1.0, n - p - 1))
  return eta_hat, aic


# @njit
# def log_lik_at_location(eta0, exp_eta1, exp_eta2, eta3, eta4, l, a, y_next, distance_matrix, product_matrix, L):
#   prod = 1.0
#   a_l = a[l]
#   for l_prime in range(L):
#     d_l_lprime = distance_matrix[l, l_prime]
#     s_l_lprime = product_matrix[l, l_prime]
#     a_l_prime = a[l_prime]
#     logit_transmission_prob = eta0 - exp_eta1 * d_l_lprime / np.power(s_l_lprime, exp_eta2) + \
#       eta3 * a_l + eta4 * a_l_prime
#     transmission_prob = 1.0 - 1.0 / (1.0 + np.exp(logit_transmission_prob))
#     prod *= (1 - transmission_prob)
#   if y_next[l]:
#     return np.log(1 - prod)
#   else:
#     return np.log(prod)
