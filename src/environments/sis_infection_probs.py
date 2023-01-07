import numpy as np
from numba import njit

@njit
def expit(logit_p):
  return 1 - 1 / (1 + np.exp(logit_p))

def sis_infection_probability(a, y, eta, L, adjacency_lists, **kwargs):
  omega, s = kwargs['omega'], kwargs['s']
  z = np.random.binomial(1, omega)
  indicator = (z * s <= 0)
  a_times_indicator = np.multiply(a, indicator)

  infected_indices = np.where(y > 0)[0].astype(int)
  not_infected_indices = np.where(y == 0)[0].astype(int)

  infected_probabilities = np.zeros(L)
  infected_probabilities[not_infected_indices] = p_l(a, a_times_indicator, not_infected_indices, infected_indices, eta,
                                                     adjacency_lists, omega) 
  infected_probabilities[infected_indices] = 1 - q_l(a[infected_indices], a_times_indicator[infected_indices], eta, omega) 
  return infected_probabilities

def p_l0(a, a_times_indicator, eta, omega):
  logit_p_0 = eta[0] + eta[1] * a_times_indicator
  p_0 = expit(logit_p_0)
  return p_0

def q_l(a, a_times_indicator, eta, omega):
  logit_q = eta[5] + eta[6] * a_times_indicator
  q = expit(logit_q)
  return q

def one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega,
                        len_not_infected, product_vector):
  product_vector = []
  for l in not_infected_indices.tolist():
    # Get infected neighbors
    infected_neighbor_indices = np.intersect1d(adjacency_lists[l], infected_indices)
    a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
    logit_p_l = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator_lprime
    p_l = expit(logit_p_l)
    product_l = np.product(1 - p_l)
    product_vector = np.append(product_vector, product_l)

  return product_vector

def p_l(a, a_times_indicator, not_infected_indices, infected_indices, eta, adjacency_lists, omega):
  p_l0_ = p_l0(a[not_infected_indices], a_times_indicator[not_infected_indices], eta, omega)
  len_not_infected = len(not_infected_indices)
  initial_product_vector = np.ones(len_not_infected)
  one_minus_p_llprime_ = one_minus_p_llprime(a, a_times_indicator, not_infected_indices, infected_indices, eta,
                                             adjacency_lists, omega, len_not_infected, initial_product_vector)
  product = np.multiply(1 - p_l0_, one_minus_p_llprime_) 
  return 1 - product
