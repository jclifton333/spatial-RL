import xpress as xp
import numpy as np
import pdb


def solve_nonlinear_program(q, treatment_budget, L):
  # Specify
  problem = xp.problem()
  a = np.array([xp.var(vartype=xp.binary) for _ in range(L)], dtype=xp.npvar)
  problem.addVariable(a)
  constr = (xp.Sum([a[i] for i in range(L)]) <= treatment_budget)
  problem.setObjective(xp.user(q, a), sense=xp.maximize)
  problem.addConstraint(constr)

  # Solve
  problem.solve()
  a_solution = problem.getSolution(a)

  return a_solution


def argmaxer_nonlinear(q, evaluation_budget, treatment_budget, env):
  def q_sum(a_):
    return np.sum(q(a_))
  a = solve_nonlinear_program(q_sum, treatment_budget, env.L)
  return a
