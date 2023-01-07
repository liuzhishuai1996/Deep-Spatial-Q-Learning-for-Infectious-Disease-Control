import numpy as np
from scipy.special import expit


def central_diff_grad(f, x0, h=0.01):
  dim = len(x0)
  grad = np.zeros(dim)
  for i in range(dim):
    e_i = np.zeros(dim)
    e_i[i] = h
    grad[i] = (f(x0 + e_i) - f(x0 - e_i)) / 2*h
  return grad


def central_diff_hess(f, x0, h=0.01):
  dim = len(x0)
  hess = np.zeros((dim, dim))
  for i in range(dim):
    e_i = np.zeros(dim)
    e_i[i] = h
    for j in range(dim):
      e_j = np.zeros(dim)
      e_j[j] = h
      if i == j:
        hess[i, j] = (-f(x0 + 2*e_i) + 16*f(x0 + e_i) - 30*f(x0) + 16*f(x0 - e_i) - f(x0 - 2*e_i)) / \
                     (12*h**2)
      else:
        hess[i, j] = (f(x0 + e_i + e_j) - f(x0 + e_i - e_j) - f(x0 - e_i + e_j) + f(x0 - e_i - e_j)) / \
                     (4*h**2)
  return hess


def logit_gradient(x, y, beta):
  x_dot_beta = np.dot(x, beta)
  expit_x_dot_beta = expit(x_dot_beta)
  expit_derivative_ = expit_derivative(x_dot_beta)
  success_component = y / expit_x_dot_beta * expit_derivative_ * x
  failure_component = (1 - y) / (1 - expit_x_dot_beta) * -expit_derivative_ * x
  return success_component + failure_component


def logit_hessian(x, beta):
  outer_ = np.outer(x, x)
  expit_x_dot_beta = expit(np.dot(x, beta))
  return outer_ * expit_x_dot_beta * (1 - expit_x_dot_beta)


def expit_derivative(x):
  """
  For delta method estimate of mb variance.
  :param x:
  :return:
  """
  return np.exp(-x) / (1 + np.exp(-x))**2