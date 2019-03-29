"""
Solve a quadratic binary program with treatment_budget constraint.
For taking argmax of quadratic approximation to Q function.


ref: http://www.gurobi.com/documentation/7.0/examples/dense_py.html
"""
import numpy as np
import pdb
from gurobipy import *


def qp_max(M, r, budget):
  """
  max x'Mx + r
  s.t. 1'x = budget

  :param M:
  :param r:
  :param budget:
  :return:
  """
  pdb.set_trace()

  model = Model('qip')
  model.setParam('OutputFlag', False)
  L = M.shape[0]

  # Define decision variables
  vars = []
  for j in range(L):
    vars.append(model.addVar(vtype=GRB.BINARY))

  # Define objective
  obj = QuadExpr()
  for i in range(L):
    for j in range(L):
      obj += M[i,j]*vars[i]*vars[j]
  obj += r
  model.setObjective(obj)

  # Define constraint
  constr_expr = LinExpr()
  constr_expr.addTerms([1.0]*L, vars)
  model.addConstr(constr_expr == budget)

  # Optimize
  model.optimize()

  return np.array([v.X for v in vars])

