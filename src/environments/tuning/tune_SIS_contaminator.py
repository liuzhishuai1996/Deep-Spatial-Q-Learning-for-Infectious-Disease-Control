"""
To see the effects of model misspecification on MB vs. MF approaches, we examine models of the form
\epsilon * SIS_transition_probs + (1 - \epsilon) * contamination_transition_probs.

contamination_transition_probs will be output by a neural network, which is tuned as follows:

1. Estimate r(0) = MSE( \hat{p}_MB | \epsilon=0) / MSE( \hat{p}_MF | \epsilon=0 ), i.e. the relative performance of
   MB and MF one-step probability estimates when the sis model is uncontaminated, for a size 50 lattice integrated
   over 25 time steps.
2. For \epsilon = 0.25, 0.75, 1, randomly generate contamination network parameters, and estimate relative MSEs
   for each.
3. Use ?
   to find contamination parameter that minimizes \lVert r(\epsilon; \beta) - r(\epsilon) \rVert^2,
   where r(\epsilon; \beta) is the observed ratio of MSEs and r(\epsilon) is the desired ratio - a line
   with slope (2 - r(0)) so that at full maximum contamination, MF MSE is half that of MB.
"""
import pdb
# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)
import numpy as np
from src.environments import generate_network, sis
from src.environments.sis_contaminator import SIS_Contaminator
from src.estimation.optim.argmaxer_factory import argmaxer_random
from src.estimation.q_functions.q_functions import q_max_all_states
from src.estimation.optim.quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
from scipy.special import logit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

SEED = 3


def logistic(n, L, k, x0):
  return L / (1 + np.exp(-k * (n - x0)))


def parameterized_expit_parameter(k, x0, scale_neighbor_params):
  """
  logistic fn = f(n) = L / (1 + exp(-k * (n - x0));
  neighbor_params = self_params * scale_neighbor_params
  :param L:
  :param k:
  :param x0:
  :param scale_neighbor_params:
  :return:
  """
  self_param =  np.array([logistic(n, 1.0, k, x0) for n in range(8)])
  neighbor_param = np.array([np.min((1.0, p * scale_neighbor_params)) for p in self_param])
  old_index_to_new_index_mapping = {0: 4,
                                    1: 5,
                                    2: 6,
                                    3: 7,
                                    4: 0,
                                    5: 1,
                                    6: 2,
                                    7: 3}
  untransformed_param = np.concatenate((self_param, neighbor_param))
  param = np.zeros(16)
  for i in range(8):
    param[old_index_to_new_index_mapping[i]] = untransformed_param[i]
  for i in range(8, 16):
    param[old_index_to_new_index_mapping[i-8] + 8] = untransformed_param[i]
  return param


def evaluate_parameter(k, x0, scale_neighbor_params, intercept):
  env = sis.SIS(L = 100, omega = 0.0, generate_network = generate_network.lattice)

  # Generate some data
  np.random.seed(SEED)
  dummy_action = np.append(np.ones(5), np.zeros(95))
  env.reset()
  for t in range(10):
    env.step(np.random.permutation(dummy_action))

  # Get true probabilities for gen mod parameterized by L, k, x0, scale_neighbor_params
  expit_parameter = parameterized_expit_parameter(k, x0, scale_neighbor_params)
  parameter = np.append(logit(expit_parameter), [intercept])
  contaminator = SIS_Contaminator()
  contaminator.set_weights(parameter)
  q_ = contaminator.predict_proba
  pdb.set_trace()

  # Evaluate random policy
  random_score_list = []
  for rep in range(10):
    score_list, _, _ = q_max_all_states(env, 100, 5, q_, argmaxer_random)
    random_score_list.append(np.mean(score_list))
  random_score = np.mean(random_score_list)

  # Evaluate true probs policy
  best_score_list, _, _ = q_max_all_states(env, 100, 5, q_, argmaxer_quad_approx)
  best_score = np.mean(best_score_list)

  loss = np.abs((random_score/best_score) - 0.5)
  return loss


def evaluate_parameter_wrapper(parameter_parameter):
  L = np.exp(parameter_parameter[0])
  k = parameter_parameter[1]
  x0 = parameter_parameter[2]
  scale_neighbor_params = np.exp(parameter_parameter[3])
  intercept = parameter_parameter[4]
  return evaluate_parameter(L, k, x0, scale_neighbor_params, intercept)


def tune():
  for i in range(10):
    k = np.random.gamma(1.0, 1.0)
    x0 = np.random.gamma(1.0, 1.0)
    scale_neighbor_params = np.random.random()
    intercept = -np.random.gamma(1.0, 1.0)
    print(evaluate_parameter(k, x0, scale_neighbor_params, intercept))
  return


def fit_to_omega_contaminated_sis():
  env = sis.SIS(L=300, omega=1.0, generate_network=generate_network.lattice)
  # Generate some data
  np.random.seed(SEED)
  dummy_action = np.append(np.ones(5), np.zeros(95))
  env.reset()
  for t in range(10):
    env.step(np.random.permutation(dummy_action))

  # Fit logit
  clf = LogisticRegression()
  X = np.vstack(env.X)
  y = np.hstack(env.y).astype(float)
  clf.fit(X, y)
  print(clf.coef_)
  print(clf.intercept_)


if __name__ == '__main__':
  fit_to_omega_contaminated_sis()
