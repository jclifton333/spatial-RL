"""
Solve a quadratic binary program with treatment_budget constraint.
For taking argmax of quadratic approximation to Q function.


ref: http://www.gurobi.com/documentation/7.0/examples/dense_py.html
"""
import numpy as np
from gurobipy import *


def qp_max(M, b, r, budget):
  """
  max x'Mx + x'b + r
  s.t. 1'x = budget

  :param M:
  :param b:
  :param r:
  :param budget:
  :return:
  """

  model = Model('qip')
  L = len(b)

  # Define decision variables
  vars = []
  for j in range(L):
    vars.append(model.addVar(vtype=GRB.BINARY))

  # Define objective
  obj = QuadExpr()
  for i in range(L):
    for j in range(L):
      obj += M[i,j]*vars[i]*vars[j]
  for i in range(L):
    obj += b[i]*vars[i]
  obj += r
  model.setObjective(obj)

  # Define constraint
  constr_expr = LinExpr()
  constr_expr.addTerms(1.0, vars)
  model.addConstr(constr_expr == budget)

  # Optimize
  model.optimize()

  return


if __name__ == '__main__':
  M = np.eye(2)
  b = np.ones(2)
  r = 0
  budget = 1
  qp_max(M, b, r, budget)