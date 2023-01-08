"""
From RSS paper, pdf pg. 13.

Decision rule construction (suppressing dependnce on time t)
  Can treat c locations
  E is space of parameters \eta
  R(s, a; eta) is a vector of priority scores (one per location), assuming locations j where
    a_j = 1 are going to be treated
  for nonnegative integers m, define
    U_l(s, a; eta, m) = 1 if R_l(s, a; eta) >= R_(m)(s, a; eta) ( i.e. mth order stat)
                        0 o.w.
  take k <= c
  define d^(1)(s; eta) to be the binary vector that selects floor(c / k) highest-priority locs
    Let w^(1) denote d^(1)(s; eta)
  Recursively, for j=2,...,k
    w^(j) = d^(j)(s; eta)
    delta_j = floor(j*c / k) - floor((j-1)*c/k)
    d^(j) = U(s, w^(j-1); eta, delta_j) + w^(j-1)
  Final decision rule is d^(k)(s; eta)

  C^T(d; beta, theta) is expected cumulative value under decision rule d, against model parametrized by theta, beta,
  up to horizon T.

  Need to take argmax of C^T(d; \hat{beta}, \hat{theta}) over class of policies described above.

Inputs
  T
  S
  eta_0
  f(y | s, a, beta)
  g(s | s, a, theta)
  { alpha_j }_j>=1
  { zeta_j }_j>=1
  tol > 0

set k = 1, \tilde{S} = S
draw Z^k from unif(-1, 1)^d

while alpha_k >= tol
  from m = 1, ..., T-1
    set A^t+m = d(S^t+m, eta^k + zeta_k Z^k)
    draw S^t+m+1 ~ g(s^t+m+1 | s^t+m, a^t+m; theta)
    draw Y^t+m ~ f(y^t+m | s^t+m, a^t+m; beta)
    set \tilde{A}^t+m
    draw \tilde{Y}^t+m...
    draw \tilde{S}^t+m+1...
  set eta^k+1 = G_E [ eta^k + alpha_k / (2 zeta_k) (Z^k 1^T_L (Y^t+T-1 - \tilde{Y}^t+T-1))]
  set k = k + 1
output eta_k

where G_E(x) is the projection of x onto the parameter space E (where eta lives)
"""
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

import matplotlib.pyplot as plt
import pdb
import numpy as np
import copy

import src.environments.sis_infection_probs as sis_inf_probs
import src.environments.gravity_infection_probs as ebola_inf_probs
from numba import njit, jit
from src.estimation.model_based.sis.estimate_sis_parameters import fit_infection_prob_model
from bayes_opt import BayesianOptimization
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model


def R(env, s, a, y, infection_probs_predictor, infection_probs_kwargs, transmission_prob_predictor,
      transmission_probs_kwargs, data_depth, eta, beta):
  """
  Linear priority score function.

  """
  priority_features = features_for_priority_score(env, s, a, y, infection_probs_predictor, infection_probs_kwargs,
                                                  transmission_prob_predictor, transmission_probs_kwargs, data_depth,
                                                  beta)
  return np.dot(priority_features, eta)


def update_eta(eta, alpha, zeta, z, y, y_tilde):
  ones = np.ones(len(y))
  second_term = z * np.dot(ones, y - y_tilde)
  new_eta = eta + alpha / (2 * zeta) * second_term
  new_eta_norm = np.linalg.norm(new_eta)
  new_eta /= np.max((1.0, new_eta_norm))  # Project onto unit sphere.
  return new_eta


# @njit
def U(priority_scores, m):
  """

  :param priority_scores: Corresponds to R above.
  :param m: Integer >= 1.
  :return:
  """
  # priority_scores_mth_order_stat = np.argsort(priority_scores)[int(m)]  # ToDo: Optimize?
  m = int(m-1)
  priority_scores_mth_order_stat = np.partition(priority_scores.flatten(), m)[m]
  U = priority_scores >= priority_scores_mth_order_stat
  return U


def decision_rule(env, s, a, y, infection_probs_predictor, infection_probs_kwargs, transmission_probs_predictor,
                  transmission_probs_kwargs, eta, beta, k, treatment_budget, priority_scores):

  d = np.zeros(len(priority_scores))
  if k == 1:
    d[np.argsort(-priority_scores)[:treatment_budget]] = 1
  # ToDo: Make sure this is working as intended...
  else:
    floor_c_by_k = int(np.floor(treatment_budget / k))
    d[np.argsort(-priority_scores)[:floor_c_by_k]] = 1
    deltas = [np.floor(j * treatment_budget / k) - np.floor((j - 1) * treatment_budget / k) for j in range(1, k)]
    for j in range(1, k):
      w = d
      delta_j = deltas[j-1]
      priority_scores = R(env, s, w, y, infection_probs_predictor, infection_probs_kwargs, transmission_probs_predictor,
                          transmission_probs_kwargs, env.data_depth, eta, beta)
      d = U(priority_scores, delta_j) + w
  return d


def update_alpha_and_zeta(alpha, zeta, j, rho, tau):
  """

  :param alpha:
  :param zeta:
  :param j:
  :param tau: Tuning parameter chosen with double bootstrap (?)
  :param rho: Tuning parameter chosen with double bootstrap (?)
  :return:
  """
  new_alpha = np.power(tau / (rho + j + 1), 1.25)
  new_zeta = 100.0 / (j + 1)
  return new_alpha, new_zeta


def roll_out_candidate_policy(T, s, a, y, beta, eta, treatment_budget, k, env, infection_probs_predictor,
                              infection_probs_kwargs, transmission_probs_predictor, transmission_probs_kwargs,
                              data_depth, number_of_steps_ahead=0, monte_carlo_reps=10, gamma=0.9):
  scores = []
  for _ in range(monte_carlo_reps):
    s_tpm = s
    y_tpm = y
    a_tpm = a
    for m in range(T):
      infection_probs_kwargs['s'] = s_tpm
      transmission_probs_kwargs['s'] = s_tpm
      transmission_probs_kwargs['y'] = y_tpm
      priority_score = R(env, s_tpm, a_tpm, y_tpm, infection_probs_predictor, infection_probs_kwargs,
                         transmission_probs_predictor, transmission_probs_kwargs, data_depth, eta, beta)
      a_tpm = decision_rule(env, s_tpm, a_tpm, y_tpm, infection_probs_predictor, infection_probs_kwargs,
                            transmission_probs_predictor, transmission_probs_kwargs, eta,
                            beta, k, treatment_budget, priority_score)
      infection_probs = infection_probs_predictor(a_tpm, y_tpm, beta, env.L, env.adjacency_list,
                                                  **infection_probs_kwargs)
      y_tpm = np.random.binomial(n=1, p=infection_probs)
      s_tpm = env.update_state(s_tpm)
      if m >= number_of_steps_ahead:  # In case we want returns starting a few steps in the future; used in continuation policies
        r_tpm = (gamma**m) * -np.sum(y_tpm)
        scores.append(r_tpm)
  return np.mean(scores)


def gp_opt_for_policy_search(T, s, y, beta, eta_init, treatment_budget, k, env, infection_probs_predictor,
                             infection_probs_kwargs, transmission_probs_predictor, transmission_probs_kwargs,
                             data_depth, n_rep_per_gp_opt_iteration=10):

  # Objective is mean score over n_rep_per_gp_opt_iteration MC replicates
  def objective(eta1, eta2, eta3):
    eta = np.array([eta1, eta2, eta3])
    a_dummy = np.zeros(env.L)
    score = roll_out_candidate_policy(T, s, a_dummy, y, beta, eta, treatment_budget, k, env,
                                      infection_probs_predictor, infection_probs_kwargs, transmission_probs_predictor,
                                      transmission_probs_kwargs, data_depth,
                                      monte_carlo_reps=n_rep_per_gp_opt_iteration, gamma=0.9)
    return score


  ETA_BOUNDS = (0.0, np.power(1, -1/3))
  explore_ = {'eta1': [eta_init[0]], 'eta2': [eta_init[1]], 'eta3': [eta_init[2]]}
  bounds = {'eta1': ETA_BOUNDS, 'eta2': ETA_BOUNDS, 'eta3': ETA_BOUNDS}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_) 
  bo.maximize(init_points=10, n_iter=10, alpha=1e-4)
  # bo.maximize()
  best_param = bo.res['max']['max_params']
  best_params = [best_param['eta1'], best_param['eta2'], best_param['eta3']]

  return best_params


def stochastic_approximation_for_policy_search(T, s, y, beta, eta, alpha, zeta, tol, maxiter, dimension,
                                               treatment_budget, k, feature_function, env, infection_probs_predictor,
                                               infection_probs_kwargs, transmission_prob_predictor,
                                               transmission_probs_kwargs, data_depth, rho, tau):
  """

  :param tau: stepsize hyperparameters
  :param rho: stepsize hyperparameters
  :param data_depth:
  :param transmission_prob_predictor:
  :param infection_probs_predictor:
  :param env:
  :param T:
  :param s:
  :param y:
  :param eta:
  :param alpha:
  :param zeta:
  :param tol:
  :param maxiter:
  :param dimension: dimension of policy parameter
  :param treatment_budget:
  :param k: number of locations to change during decision rule iterations
  :param feature_function:
  :return:
  """
  DIFF_TOL = 0.0001

  it = 0
  a_dummy = np.zeros(env.L)  
  diff = float('inf')
  alpha, zeta = update_alpha_and_zeta(alpha, zeta, 0, rho, tau)
  while alpha > tol and it < maxiter and diff > DIFF_TOL:
    z = np.random.random(size=dimension)
    s_tpm = s
    y_tpm = y
    # x_tpm = feature_function(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor,
    #                          data_depth, beta)

    s_tpm_tilde = s
    y_tpm_tilde = y
    # x_tpm_tilde = feature_function(env, s_tpm_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
    #                                transmission_prob_predictor, data_depth, beta)

    s_tpmp1 = s_tpm
    s_tpmp1_tilde = s_tpm

    for m in range(T):
      # Plus perturbation
      eta_plus = eta + zeta * z
      priority_score_plus = R(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, infection_probs_kwargs,
                              transmission_prob_predictor, transmission_probs_kwargs, data_depth, eta_plus, beta)
      a_tpm = decision_rule(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, infection_probs_kwargs,
                            transmission_prob_predictor, transmission_probs_kwargs, eta, beta, k, treatment_budget,
                            priority_score_plus)
      infection_probs = infection_probs_predictor(a_tpm, y_tpm, beta, env.L, env.adjacency_list,
                                                  **infection_probs_kwargs)
      y_tpm = np.random.binomial(n=1, p=infection_probs)
      # x_tpm = feature_function(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor,
      #                          transmission_prob_predictor, data_depth, beta)

      # Minus perturbation
      eta_minus = eta - zeta * z
      priority_score_minus = R(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, infection_probs_kwargs,
                               transmission_prob_predictor, transmission_probs_kwargs, data_depth, eta_minus, beta)
      a_tpm_tilde = decision_rule(env, s_tpm_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
                                  infection_probs_kwargs, transmission_prob_predictor, transmission_probs_kwargs, eta,
                                  beta, k, treatment_budget, priority_score_minus)
      infection_probs_tilde = infection_probs_predictor(a_tpm_tilde, y_tpm_tilde, beta, env.L, env.adjacency_list,
                                                        **infection_probs_kwargs)
      y_tpm_tilde = np.random.binomial(n=1, p=infection_probs_tilde)
      # x_tpm_tilde = feature_function(env, s_tpmp1_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
      #                                transmission_prob_predictor, data_depth, beta)

      # Update states
      s_tpm = s_tpmp1
      s_tpm_tilde = s_tpmp1_tilde

    new_eta = update_eta(eta, alpha, zeta, z, y_tpm, y_tpm_tilde)
    diff = np.linalg.norm(eta - new_eta) / np.max((0.001, np.linalg.norm(eta)))
    eta = copy.copy(new_eta)

    it += 1
    alpha, zeta = update_alpha_and_zeta(alpha, zeta, it, rho, tau)
    # print('it: {}\nalpha: {}\nzeta: {}\neta: {}'.format(it, alpha, zeta, eta))
  # print('number of iterations: {}'.format(it))
  return eta


"""
Implementing priority score below. See pdf pg. 15.
"""


# @njit
def psi(infected_locations, predicted_infection_probs, lambda_, transmission_probabilities, data_depth):
  """
  Different from 'psi' for env-specific features!

  :param transmission_probabilities:
  :param infected_locations:
  :param predicted_infection_probs:
  :param lambda_: LxL matrix
  :param transmission_proba  :param m_hat: LxL matrix of estimated transmission probabilities under estimated modelbilities:

  :param data_depth: vector of [c^l] in paper
  :return:
  """
  psi_1 = predicted_infection_probs
  len_psi_1 = len(psi_1)

  # Compute multiplier, not sure this is right
  transmission_probabilities_inf = transmission_probabilities[:, infected_locations[0]]
  lambda_inf = lambda_[:, infected_locations[0]]
  transmission_probs_times_lambda_inf = np.multiply(transmission_probabilities_inf, lambda_inf)
  multiplier = np.dot(transmission_probs_times_lambda_inf, 1 - predicted_infection_probs[infected_locations])

  # psi_2 = np.multiply(psi_1, multiplier)
  # psi_3 = np.multiply(psi_1, data_depth)

  psi_2 = np.zeros(len_psi_1)
  psi_3 = np.zeros(len_psi_1)
  for i in range(len_psi_1):
    psi_2[i] = psi_1[i]*multiplier[i]
    psi_3[i] = psi_1[i]*data_depth[i]
  return psi_1, psi_2, psi_3


@njit
# @jit
def phi(not_infected_locations, lambda_, transmission_probabilities, psi_1, psi_2, data_depth, len_not_inf, L):
  lambda_inf = lambda_[:, not_infected_locations]
  transmission_probabilities_not_inf = transmission_probabilities[:, not_infected_locations]

  psi_1_not_inf = psi_1[not_infected_locations]
  psi_2_not_inf = psi_2[not_infected_locations]
  data_depth_not_inf = data_depth[not_infected_locations]

  # phi_1 = np.dot(lambda_inf, psi_1_not_inf)
  # phi_2 = np.dot(transmission_probabilities_not_inf, psi_2_not_inf)
  # phi_3 = np.dot(transmission_probabilities_not_inf, data_depth_not_inf)
  phi_1 = [0.0 for _ in range(L)]
  phi_2 = [0.0 for _ in range(L)]
  phi_3 = [0.0 for _ in range(L)]

  for i in range(L):
    for j in range(len_not_inf):
      phi_2[i] += transmission_probabilities_not_inf[i, j]*psi_2_not_inf[j]
      phi_3[i] += transmission_probabilities_not_inf[i, j]*data_depth_not_inf[j]
    for j in range(len_not_inf):
      phi_1[i] += lambda_inf[i, j]*psi_1_not_inf[j]

  # phi = np.column_stack((phi_1, phi_2, phi_3))
  return phi_1, phi_2, phi_3


def features_for_priority_score(env, s, a, y, infection_probs_predictor, infection_probs_kwargs,
                                transmission_prob_predictor, transmission_probs_kwargs, data_depth, beta):
  lambda_ = env.lambda_

  # Get predicted probabilities
  predicted_infection_probs = infection_probs_predictor(a, y, beta, env.L, env.adjacency_list, **infection_probs_kwargs)
  transmission_probabilities = transmission_prob_predictor(a, beta, env.L, **transmission_probs_kwargs)

  # Get infection status-specific features
  infected_locations = np.where(y == 1)
  not_infected_locations = np.where(y == 0)
  psi_1, psi_2, psi_3 = psi(infected_locations, predicted_infection_probs, lambda_, transmission_probabilities,
                            data_depth)
  len_not_inf = len(not_infected_locations[0])
  phi_1, phi_2, phi_3 = \
    phi(not_infected_locations[0], lambda_, transmission_probabilities, psi_1, psi_2, data_depth, len_not_inf, env.L)
  phi_ = np.column_stack((phi_1, phi_2, phi_3))

  # Collect features
  priority_score_features = np.zeros((env.L, 3))
  psi_not_inf = np.column_stack((psi_1, psi_2, psi_3))[not_infected_locations, :]
  phi_inf = phi_[infected_locations, :]
  priority_score_features[not_infected_locations, :] = psi_not_inf
  priority_score_features[infected_locations, :] = phi_inf

  return priority_score_features


def policy_parameter(env, time_horizon, gen_model_posterior, initial_policy_parameter, initial_alpha, initial_zeta,
                    treatment_budget, rho, tau, infection_probs_predictor, infection_probs_kwargs,
                    transmission_probs_predictor, transmission_probs_kwargs,
                    tol=1e-3, maxiter=100, feature_function=features_for_priority_score, k=1,
                    method='bayes_opt'):

  dimension = len(initial_policy_parameter)
  beta_tilde = gen_model_posterior()

  if method == 'stochastic_approximation':
    policy_parameter = stochastic_approximation_for_policy_search(time_horizon, env.current_state, env.current_infected, beta_tilde,
                                                                  initial_policy_parameter,
                                                                  initial_alpha, initial_zeta, tol, maxiter,
                                                                  dimension, treatment_budget, k, feature_function, env,
                                                                  infection_probs_predictor, infection_probs_kwargs,
                                                                  transmission_probs_predictor,
                                                                  transmission_probs_kwargs, env.data_depth, rho, tau)
  elif method == 'bayes_opt':
    policy_parameter = gp_opt_for_policy_search(time_horizon, env.current_state, env.current_infected, beta_tilde,
                                                initial_policy_parameter, treatment_budget, k, env,
                                                infection_probs_predictor, infection_probs_kwargs,
                                                transmission_probs_predictor, transmission_probs_kwargs, env.data_depth,
                                                n_rep_per_gp_opt_iteration=50)

  return policy_parameter, beta_tilde


def policy_search(env, time_horizon, gen_model_posterior, initial_policy_parameter, initial_alpha, initial_zeta,
                  treatment_budget, rho, tau, tol=1e-3, maxiter=100, feature_function=features_for_priority_score, k=1,
                  method='bayes_opt', oracle=False):
  """
  Alg 1 on pg 10 of Nick's WNS paper; referring to parameter of transition model as 'beta', instead of 'eta'
  as in QL draft and the rest of this source code

  :param tau: SA stepsize hyperparameter
  :param rho: SA stepsize hyperparameter
  :param treatment_budget:
  :param infection_probs_predictor:
  :param transmission_probs_predictor:
  :param feature_function:
  :param maxiter:
  :param tol:
  :param initial_zeta:
  :param initial_alpha:
  :param env:
  :param time_horizon:
  :param gen_model_posterior: function that returns draws from conf dbn over gen model parameter
  :param initial_policy_parameter:
  :param k: number of locations to change during decision rule iterations
  :param method: either 'bayes_opt' or 'stochastic_approximation'
  :return:
  """
  if env.__class__.__name__ == 'SIS':
    infection_probs_kwargs = {'s': env.current_state, 'omega': 0.0}
    transmission_probs_kwargs = {'adjacency_matrix': env.adjacency_matrix}
    if oracle:
      infection_probs_predictor = sis_inf_probs.sis_infection_probability_oracle_contaminated
      infection_probs_kwargs['epsilon'] = env.epsilon
      infection_probs_kwargs['contaminator'] = env.contaminator
      infection_probs_kwargs['feature_function'] = env.binary_psi
      if env.epsilon > 0:
        transmission_probs_kwargs['epsilon'] = env.epsilon
        transmission_probs_kwargs['contaminator'] = env.contaminator
        transmission_probs_kwargs['feature_function'] = env.binary_psi
        transmission_probs_kwargs['s'] = env.current_state
        transmission_probs_kwargs['y'] = env.current_infected
        transmission_probs_predictor = sis_inf_probs.get_all_oracle_contaminated_sis_transmission_probs
      else:
        transmission_probs_predictor = sis_inf_probs.get_all_sis_transmission_probs_omega0
    else:
      infection_probs_predictor = sis_inf_probs.sis_infection_probability
      transmission_probs_predictor = sis_inf_probs.get_all_sis_transmission_probs_omega0
  elif env.__class__.__name__ == 'Ebola':
    infection_probs_kwargs = {'distance_matrix': env.DISTANCE_MATRIX, 'susceptibility': env.SUSCEPTIBILITY,
                              'adjacency_matrix': env.ADJACENCY_MATRIX, 'product_matrix': env.product_matrix,
                              'x': None}
    transmission_probs_kwargs = {'distance_matrix': env.DISTANCE_MATRIX, 'susceptibility': env.SUSCEPTIBILITY,
                              'adjacency_matrix': env.ADJACENCY_MATRIX, 'product_matrix': env.product_matrix,
                              'x': None}
    infection_probs_predictor = ebola_inf_probs.ebola_infection_probs
    transmission_probs_predictor = ebola_inf_probs.get_all_gravity_transmission_probs

  policy_parameter_, beta_tilde = policy_parameter(env, time_horizon, gen_model_posterior, initial_policy_parameter,
                                                   initial_alpha,
                                                   initial_zeta, treatment_budget, rho, tau,
                                                   infection_probs_predictor, infection_probs_kwargs,
                                                   transmission_probs_predictor,
                                                   transmission_probs_kwargs,
                                                   tol=1e-3, maxiter=100,
                                                   feature_function=features_for_priority_score, k=1, method='bayes_opt')

  # Get priority function features
  a_for_transmission_probs = np.zeros(env.L)  # ToDo: Check which action is used to get transmission probs

  # ToDo: distinguish between env.pairwise_distances and Gravity.DISTANCE_MATRIX !
  # transmission_probabilities = transmission_probs_predictor(a_for_transmission_probs, beta_tilde, env.L,
  #                                                           env.adjacency_matrix)

  # infected_locations = np.where(env.current_infected == 1)
  # predicted_infection_probs = infection_probs_predictor(a_for_transmission_probs, env.current_infected,
  #                                                       env.current_state, beta_tilde, 0.0, env.L,
  #                                                       env.adjacency_list)

  features = feature_function(env, env.current_state, a_for_transmission_probs, env.current_infected,
                              infection_probs_predictor, infection_probs_kwargs, transmission_probs_predictor,
                              transmission_probs_kwargs, env.data_depth, beta_tilde)

  priority_scores = np.dot(features, policy_parameter_)
  a_ix = np.argsort(-priority_scores)[:treatment_budget]
  a = np.zeros(env.L)
  a[a_ix] = 1
  return a, policy_parameter_


def policy_parameter_wrapper(**kwargs):
  """
  Helper for policies that depend on policy search.

  :param kwargs:
  :return:
  """
  env, remaining_time_horizon, treatment_budget, initial_policy_parameter = \
    kwargs['env'], kwargs['planning_depth'], kwargs['treatment_budget'], kwargs['initial_policy_parameter']

  if env.__class__.__name__ == "SIS":
    beta_mean, _ = fit_infection_prob_model(env, None)
    beta_cov = env.mb_covariance(beta_mean)

    def gen_model_posterior():
      beta_tilde = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)
      return beta_tilde
  elif env.__class__.__name__ == "Gravity":
    # beta_mean = fit_ebola_transition_model(env)
    # beta_cov = env.mb_covariance(beta_mean)
    def gen_model_posterior():
      beta_tilde = fit_ebola_transition_model(env, bootstrap=True)
      return beta_tilde

  # Settings
  if initial_policy_parameter is None:
    initial_policy_parameter = np.ones(3) * 0.5
  initial_alpha = initial_zeta = None
  # remaining_time_horizon = T - env.T

  # ToDo: These were tuned using bayes optimization on 10 mc replicates from posterior obtained after 15 steps of random
  # ToDo: policy; may be improved...
  rho = 3.20
  tau = 0.76

  # ToDo: Write function that dooes this
  if env.__class__.__name__ == 'SIS':
    infection_probs_kwargs = {'s': np.zeros(env.L), 'omega': 0.0}
    transmission_probs_kwargs = {'adjacency_matrix': env.adjacency_matrix}
    infection_probs_predictor = sis_inf_probs.sis_infection_probability
    transmission_probs_predictor = sis_inf_probs.get_all_sis_transmission_probs_omega0
  elif env.__class__.__name__ == 'Ebola':
    infection_probs_kwargs = {'distance_matrix': env.DISTANCE_MATRIX, 'susceptibility': env.SUSCEPTIBILITY,
                              'adjacency_matrix': env.ADJACENCY_MATRIX, 'product_matrix': env.product_matrix,
                              'x': None}
    transmission_probs_kwargs = infection_probs_kwargs
    infection_probs_predictor = ebola_inf_probs.ebola_infection_probs
    transmission_probs_predictor = ebola_inf_probs.get_all_gravity_transmission_probs

  policy_parameter_, beta_tilde = policy_parameter(env, remaining_time_horizon, gen_model_posterior, initial_policy_parameter,
                                                   initial_alpha, initial_zeta, treatment_budget, rho, tau,
                                                   infection_probs_predictor, infection_probs_kwargs, transmission_probs_predictor,
                                                   transmission_probs_kwargs, tol=1e-3,
                                                   maxiter=100, feature_function=features_for_priority_score, k=1,
                                                   method='bayes_opt')
  return policy_parameter_, beta_mean


def oracle_policy_search_policy(**kwargs):
  env, remaining_time_horizon, treatment_budget, initial_policy_parameter = \
    kwargs['env'], kwargs['planning_depth'], kwargs['treatment_budget'], kwargs['initial_policy_parameter']

  def gen_model_posterior():
    return env.ETA

  # Settings
  if initial_policy_parameter is None:
    initial_policy_parameter = np.ones(3) * 0.5
  initial_alpha = initial_zeta = None
  # remaining_time_horizon = T - env.T

  # remaining_time_horizon = np.min((remaining_time_horizon, 3))

  # ToDo: These were tuned using bayes optimization on 10 mc replicates from posterior obtained after 15 steps of random
  # ToDo: policy; may be improved...
  rho = 3.20
  tau = 0.76
  a, policy_parameter = policy_search(env, remaining_time_horizon, gen_model_posterior, initial_policy_parameter,
                                      initial_alpha, initial_zeta, treatment_budget, rho, tau, tol=1e-3,
                                      maxiter=100, feature_function=features_for_priority_score, k=1,
                                      method='bayes_opt', oracle=True)
  return a, {'initial_policy_parameter': policy_parameter}


def policy_search_policy(**kwargs):
  env, remaining_time_horizon, treatment_budget, initial_policy_parameter = \
    kwargs['env'], kwargs['planning_depth'], kwargs['treatment_budget'], kwargs['initial_policy_parameter']

  if env.__class__.__name__ == "SIS":
    beta_mean, _ = fit_infection_prob_model(env, None)
    beta_cov = env.mb_covariance(beta_mean)

    def gen_model_posterior():
      beta_tilde = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)
      return beta_tilde
  elif env.__class__.__name__ == "Ebola":
    # beta_mean = fit_ebola_transition_model(env)
    # beta_cov = env.mb_covariance(beta_mean)
    def gen_model_posterior():
      beta_tilde = fit_ebola_transition_model(env, bootstrap=True)[0]
      return beta_tilde

  # Settings
  if initial_policy_parameter is None:
    initial_policy_parameter = np.ones(3) * 0.5
  initial_alpha = initial_zeta = None
  # remaining_time_horizon = T - env.T

  # ToDo: These were tuned using bayes optimization on 10 mc replicates from posterior obtained after 15 steps of random
  # ToDo: policy; may be improved...
  rho = 3.20
  tau = 0.76
  a, policy_parameter = policy_search(env, remaining_time_horizon, gen_model_posterior, initial_policy_parameter,
                                      initial_alpha, initial_zeta, treatment_budget, rho, tau, tol=1e-3,
                                      maxiter=100, feature_function=features_for_priority_score, k=1,
                                      method='bayes_opt')
  return a, {'initial_policy_parameter': policy_parameter}

