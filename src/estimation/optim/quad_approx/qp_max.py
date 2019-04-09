"""
Solve a quadratic binary program with treatment_budget constraint.
For taking argmax of quadratic approximation to Q function.


ref: http://www.gurobi.com/documentation/7.0/examples/dense_py.html
"""
import numpy as np
import pdb
try:
  from gurobipy import *
  GUROBI = True
except ImportError:
  from cvxopt import matrix, solvers
  GUROBI = False




def qp_max(M, r, budget):
  """
  max x'Mx + r
  s.t. 1'x = budget

  :param M:
  :param r:
  :param budget:
  :return:
  """
  if GUROBI:
    return qp_max_gurobi(M, r, budget)
  else:
    return qp_max_cvx(M, r, budget)


def qp_max_gurobi(M, r, budget):
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


def qp_max_cvx(M, r, budget):
  """
  CVX qp notation:
    min 0.5 x^T P x + q^T x
    s.t. Gx <= h
         Ax = b

  :param M:
  :param r:
  :param budget:
  :return:
  """
  P = matrix(2*M)
  A = matrix(np.ones(M.shape[0]))
  b = matrix([budget])

