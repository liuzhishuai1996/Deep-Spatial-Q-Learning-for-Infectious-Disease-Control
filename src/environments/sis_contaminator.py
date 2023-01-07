import pdb
import numpy as np
from scipy.special import expit


class SIS_Contaminator(object):
  def __init__(self):
    self.weights = None

  def set_weights(self, new_weights):
    self.weights = new_weights

  def predict_proba(self, X):
    logit_p = self.get_logit(X)
    p = expit(logit_p)
    return p

  def get_logit(self, X):
    if self.weights is None:
      self.weights = np.random.normal(size=(X.shape[1] + 1))
    logit_p = np.dot(np.column_stack((X, np.ones(X.shape[0]))), self.weights)
    return logit_p

  def get_neighbor_logit(self, X_neighbor):
    if self.weights is None:
      self.weights = np.random.normal(size=(X_neighbor.shape[1] + 1))
    logit_p = np.dot(X_neighbor, self.weights[8:-1])
    return logit_p

  def get_neighbor_contribution(self, X, X_neighbor):
    logit_ = self.get_logit(X)
    neighbor_logit_ = self.get_neighbor_logit(X_neighbor)
    neighbor_contrib = np.exp(neighbor_logit_) / (1 + np.exp(logit_))
    return neighbor_contrib


def recoding_mapping(contaminator_coef):
  """
  Contaminator was tuned with nodes encoded as (1*s + 2*a + 4*(1-y)), whereas nodes are now encoded as
  (1*s + 2*a + 4*y).  This transforms a contamination model parameter with the original encoding to the new encoding.

  :param contaminator_coef: length-17 contamination model parameter.
  :return:
  """
  new_contaminator_coef = np.zeros(17)
  new_encoding_action_indices = [2, 3, 10, 11]
  old_index_to_new_index_mapping = {0: 4,
                                    1: 5,
                                    2: 6,
                                    3: 7,
                                    4: 0,
                                    5: 1,
                                    6: 2,
                                    7: 3}
  for i in range(8):
    new_contaminator_coef[old_index_to_new_index_mapping[i]] = contaminator_coef[i]
  for i in range(8, 16):
    new_contaminator_coef[old_index_to_new_index_mapping[i-8] + 8] = contaminator_coef[i]
  new_contaminator_coef[-1] = contaminator_coef[-1]
  new_contaminator_coef[new_encoding_action_indices] = \
    -np.abs(new_contaminator_coef[new_encoding_action_indices]) - (2.0 + np.random.random())
  return new_contaminator_coef