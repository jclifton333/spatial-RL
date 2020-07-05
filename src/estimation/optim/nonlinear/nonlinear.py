import xpress as xp


def solve_nonlinear_program(q, treatment_budget, L):
  # Specify
  problem = xp.problem()
  a = [xp.var(xp.integer) for _ in range(L)]
  problem.addVariable(a)
  constr = (xp.Sum(a) == treatment_budget)
  problem.addConstraint(constr)
  problem.setObjective(q(a), sense=xp.maximize)

  # Solve
  problem.solve()
  a_solution = problem.getSolution(a)

  return a_solution


def argmaxer_nonlinear(q, evaluation_budget, treatment_budget, env):
  a = solve_nonlinear_program(q, treatment_budget, env.L)
  return a
