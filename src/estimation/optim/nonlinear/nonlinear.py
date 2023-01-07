# import xpress as xp
import numpy as np
from ..quad_approx.fit_quad_approx import get_quadratic_program_from_q
import pdb
import math


def solve_nonlinear_program(q, treatment_budget, L):
  def obj(a):
    # return xp.Sum([a_[i]*a_[j]*q[i, j] for i in range(L) for j in range(L)])
    return a[0]*q[0,0] + a[1]*q[0,1]

  # Specify
  problem = xp.problem()
  a = [xp.var() for _ in range(L)]
  problem.addVariable(a)
  problem.addConstraint(xp.Sum([a[i] for i in range(L)]) <= treatment_budget)
  f = xp.user(obj, a[:2])
  problem.setObjective(f)

  # Solve
  problem.solve()
  a_solution = problem.getSolution(a)

  return a_solution


def argmaxer_nonlinear(q, evaluation_budget, treatment_budget, env):
  # def q_sum(a_):
  #   q_ = 0.
  #   for i in range(L):
  #     for j in range(L):
  #       q_ += a_[i] * a_[j] * q[i, j]
  #   return q_

  # ToDo: using a quadratic approximation for testing
  quadratic_parameters, _ = get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env, None)
  a = solve_nonlinear_program(quadratic_parameters, treatment_budget, env.L)
  return a
