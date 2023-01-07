import pdb
from src.estimation.q_functions.fqi import fqi
from .estimate_sis_parameters import fit_sis_transition_model
from .simulate_from_sis import simulate_from_SIS


def estimate_sis_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth, treatment_budget,
                      evaluation_budget, argmaxer, bootstrap):

  # Estimate MDP and generate data using policy = argmax q_model
  eta = fit_transition_model(env, bootstrap=bootstrap)
  print('running mb simulations')
  simulation_env = simulate_from_SIS(env, eta, planning_depth, argmaxer, evaluation_budget,
                                     treatment_budget)
  print('estimating q function')
  # Estimate optimal q-function from simulated data
  q_model = fqi(rollout_depth, gamma, simulation_env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                bootstrap=False)

  return q_model
