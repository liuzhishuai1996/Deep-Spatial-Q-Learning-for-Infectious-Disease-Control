"""
Gradients for (log) likelihood of sis generative model.
State transitions are parameterized by beta = [beta_0, beta_1].
Infection transitions are parameterized by eta = [eta_0, ..., eta_6].

eta_p0: eta_0, eta_1
eta_p:  eta_2, eta_3, eta_4
eta_q:  eta_5, eta_6
"""
import pdb
import numpy as np
from functools import partial
from .infection_model_objective import negative_log_likelihood
from sklearn.linear_model import LinearRegression
from src.estimation.q_functions.model_fitters import SKLogit
from src.environments.sis import update_counts_for_likelihood_
from scipy.optimize import minimize
from scipy.special import logit


def fit_infection_prob_model(env, bootstrap_weights, y_next=None, indices=None):
  """

  :param env:
  :param bootstrap_weights: (env.T, env.L) - size array of bootstrap weights, or None
  :return:
  """
  if indices is None:
    X = np.vstack(env.X_raw)
    if y_next is None:
      y = np.hstack(env.y).astype(float)
    else:
      y = y_next
  else:
    X = np.vstack([x_raw[indices_at_t, :] for indices_at_t, x_raw in zip(indices, env.X_raw)])
    y = np.hstack([y_[indices_at_t] for indices_at_t, y_ in zip(indices, env.y)])

  infected_ixs = np.where(X[:, 2] == 1)
  A_infected, y_infected = X[infected_ixs, 1], y[infected_ixs]
  if bootstrap_weights is not None:
    infected_weights = bootstrap_weights.flatten()[infected_ixs]
  else:
    bootstrap_weights = None
    infected_weights = None

  if len(y_infected) > 0:
    eta_q, negative_log_lik_q = fit_q(A_infected.T, y_infected, infected_weights)
  else: 
    eta_q = np.append([logit(np.mean(y))], [0.0])
  eta_p, negative_log_lik_p = fit_p(env, bootstrap_weights, indices)
  eta = np.concatenate((eta_p, eta_q))
  negative_log_lik = negative_log_lik_p + negative_log_lik_q
  p = len(eta)
  n = X.shape[0]
  aic = p + negative_log_lik + (p**2 + p) / np.max((1.0, n - p - 1))  # Technically, AICc / 2
  return eta, aic


def fit_sis_transition_model(env, bootstrap_weights=None, y_next=None, indices=None):
  eta, _ = fit_infection_prob_model(env, bootstrap_weights, y_next=y_next, indices=indices)
  # beta = fit_state_transition_model(env)
  return eta


def fit_q(A_infected, y_infected, infected_weights):
  clf = SKLogit()
  clf.fit(A_infected, 1 - y_infected, infected_weights)
  eta_q = np.append(clf.intercept_, clf.coef_)
  negative_log_lik_q = clf.negative_log_likelihood
  return eta_q, negative_log_lik_q


def fit_p(env, bootstrap_weights, indices=None):
  if indices is not None:
    counts_for_likelihood = collect_counts_for_likelihood(env, indices)
  else:
    counts_for_likelihood = env.counts_for_likelihood
  objective = partial(negative_log_likelihood, counts_for_likelihood=counts_for_likelihood)
  res = minimize(objective, x0=env.eta[:5], method='L-BFGS-B')
  eta_p = res.x
  negative_log_lik_p = objective(eta_p)
  return eta_p, negative_log_lik_p


def collect_counts_for_likelihood(env, indices):
  """
  Collect counts for likelihood for a subset of observations, indexed by indices.

  :param env:
  :param indices: List of length T, where each element is a list of indices to keep at that time step.
  :return:
  """
  counts_for_likelihood_names = ['n_00_0', 'n_01_0', 'n_10_0', 'n_11_0', 'n_00_1', 'n_01_1', 'n_10_1', 'n_11_1',
                                 'a_0', 'a_1']
  counts_for_likelihood = {count_name: [] for count_name in counts_for_likelihood_names}
  for t, (data_block, indices_at_t) in enumerate(zip(env.X, indices)):
    counts_for_likelihood = update_counts_for_likelihood_(counts_for_likelihood, data_block, env.Y[t],
                                                          env.y[t], indices_at_t)
  return counts_for_likelihood


def estimate_variance(X, y, fitted_regression_model):
  """
  Get MSE of a simple linear regression.
  """
  y_hat = fitted_regression_model.predict(X.reshape(-1,1))
  n = len(X)
  return np.sum((y - y_hat)**2) / (n - 1)


def fit_state_transition_model(env):
  # This is not actually used in estimating model assuming omega=0.
  # ToDO: Compute online (Sherman-Woodbury)
  X = np.vstack(env.X_raw[:-1])
  X_plus = np.vstack(env.X_raw[1:])
  S, S_plus = X[:,0], X_plus[:,0]
  reg = LinearRegression(fit_intercept=False)
  reg.fit(S.reshape(-1,1), S_plus)
  beta_0_hat = reg.coef_[0]
  beta_1_hat = estimate_variance(S, S_plus, reg)
  return beta_0_hat, beta_1_hat



