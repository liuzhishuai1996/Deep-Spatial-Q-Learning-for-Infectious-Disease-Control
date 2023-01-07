# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:08:33 2018

@author: Jesse
"""
import numpy as np
from src.estimation.q_functions.q_functions import q_max_all_states
from src.estimation.q_functions.one_step import fit_one_step_predictor


def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features


def fqi(K, gamma, env, evaluation_budget, treatment_budget, classifier, regressor, argmaxer, y_next=None,
        bootstrap=True):

  if y_next is None:
    target = np.hstack(env.y).astype(float)
  else:
    target = y_next

  if K > 1:
    features = np.vstack(env.X_2)
    neighbor_order = 2  # Only supporting two-step for now
  else:
    features = np.vstack(env.X)
    neighbor_order = 1

  # Fit 1-step model
  q_one_step, _ = fit_one_step_predictor(classifier, env, None, y_next=y_next)

  q_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, q_one_step.predict_proba, argmaxer,
                                 condition_on_infection_status=True)

  # Look ahead
  for k in range(1, K + 1):
    target += gamma*q_max.flatten()
    reg = regressor()
    reg.fit(features, target)

    if k < K:
      q_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, reg.predict, argmaxer,
                                     neighbor_order=neighbor_order, condition_on_infection_status=False)
  return reg.predict


# def network_features_rollout(env, evaluation_budget, treatment_budget, regressor):
#   # target = np.sum(env.y, axis=1).astype(float)
#   target = np.sum(env.true_infection_probs, axis=1).astype(float)
#   regressor.fit(np.array(env.Phi), target)
#   Qmax, Qargmax, argmax_actions, qvals = q_max(env, evaluation_budget, treatment_budget, regressor.predict, network_features=True)
#   return argmax_actions, target
