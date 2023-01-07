import numpy as np
import src.utils.gradient as gradient
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit


def is_y_all_1_or_0(y):
    y0 = y[0]
    for element in y:
      if element == 1 - y0:
        return False
    return True


def empirical_bayes(y):
  y0 = y[0]
  n = len(y)
  expit_mean_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
  mean_ = logit(expit_mean_)
  return mean_


def empirical_bayes_coef(y, p):
  """
  For when y is all 1 or 0 and you need a coefficient vector.
  :param y:
  :param p:
  :return:
  """
  intercept_ = empirical_bayes(y)
  coef_ = np.zeros(p)
  return intercept_, coef_


class SKLogit(object):
  def __init__(self):
    self.reg = LogisticRegression()
    self.condition_on_infection = False
    self.intercept_ = None
    self.coef_ = None
    self.negative_log_likelihood = None

  def fit(self, X, y, weights):
    if is_y_all_1_or_0(y):
      self.intercept_, self.coef_ = empirical_bayes_coef(y, X.shape[1])
    else:
      self.reg.fit(X, y, sample_weight=weights)
      self.get_coef()

      # Negative log likelihood
      phat = self.reg.predict_proba(X)[:, -1]
      log_likelihood_elements = y * np.log(phat) + (1 - y) * np.log(1 - phat)
      self.negative_log_likelihood = -np.sum(log_likelihood_elements)

  def get_coef(self):
    self.intercept_ = self.reg.intercept_
    self.coef_ = self.reg.coef_

  def predict_proba(self, X):
    phat = self.reg.predict_proba(X)
    return phat[:, -1]


class SKLogit2(object):
  condition_on_infection = True

  def __init__(self):
    self.reg_= LogisticRegression()
    # self.reg_ = MLPClassifier(hidden_layer_sizes=(50,50))
    self.model_fitted = False
    self.params = None
    self.eb_prob = None
    self.aic = None

  def log_lik_gradient(self, x, y_next, infected):
    x_inf = infected * x
    x_interaction = np.concatenate(([1], x, x_inf))
    grad = gradient.logit_gradient(x_interaction, y_next, self.params)
    return grad

  def log_lik_hess(self, x, infected):
    x_inf = infected * x
    x_interaction = np.concatenate(([1], x, x_inf))
    hess = gradient.logit_hessian(x_interaction, self.params)
    return hess

  def covariance(self, X, y, infected_locations):
    n, p = X.shape[1]
    grad_outer = np.zeros((2*p, 2*p))
    hess = np.zeros((2*p, 2*p))
    for i, x, y_ in enumerate(zip(X, y)):
      infected = i in infected_locations
      grad = self.log_lik_gradient(x, y_, infected)
      hess_i = self.log_lik_hess(x, infected)
      grad_outer += np.outer(grad, grad)
      hess += hess_i
    hess_inv = np.linalg.inv(hess + 0.1*np.eye(2*p))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(n)
    return cov

  def fit(self, X, y, weights, truncate, infected_locations, not_infected_locations):
    if is_y_all_1_or_0(y):
      y0 = y[0]
      n = len(y)
      expit_intercept_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
      intercept_ = logit(expit_intercept_)
      coef_ = np.zeros(X.shape[1]*2)
      self.params = np.concatenate((intercept_, coef_))
      self.eb_prob = expit(intercept_[0])
    else:
      infection_indicator = np.array([i in infected_locations[0] for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((X, X_times_infection))
      self.X_train = X_interaction
      self.reg_.fit(X_interaction, y, sample_weight=weights)
      self.model_fitted = True
      self.params = np.concatenate(([self.reg_.intercept_, self.reg_.coef_[0]]))
    if truncate:  # ToDo: modify to reflect not-split model
      cov = self.covariance(X, y, infected_locations)
      p = X.shape[1]
      new_params = np.random.multivariate_normal(np.concatenate((self.inf_params, self.not_inf_params)), cov=cov)
      self.inf_params = new_params[:p]
      self.not_inf_params = new_params[p:]

    # Negative log likelihood
    phat = self.reg_.predict_proba(X_interaction)[:, -1]
    self.log_likelihood_elements = y * np.log(phat) + (1 - y) * np.log(1 - phat)
    negative_log_likelihood = -np.sum(self.log_likelihood_elements)
    n, p = X_interaction.shape
    self.aic = p + negative_log_likelihood + (p**2 + p) / np.max((1.0, n - p - 1))  # Technically, AICc/2

  def predict_proba(self, X, infected_locations, not_infected_locations):
    if self.model_fitted:
      infection_indicator = np.array([i in infected_locations for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((X, X_times_infection))
      phat = self.reg_.predict_proba(X_interaction)[:, -1]
    else:
      phat = self.eb_prob
    return phat

  def predict_proba_given_parameter(self, X, infected_locations, not_infected_locations, parameter):
    """
    :param X:
    :param infected_locations:
    :param not_infected_locations:
    :param parameter: array of the form [inf_intercept, inf_coef, not_inf_intercept, not_inf_coef]
    :return:
    """
    if self.model_fitted:
      infection_indicator = np.array([i in infected_locations for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((np.ones(X.shape[0]), X, X_times_infection))
      logit_phat = np.dot(X_interaction, parameter)
      phat = expit(logit_phat)
    else:
      phat = self.eb_prob
    return phat

