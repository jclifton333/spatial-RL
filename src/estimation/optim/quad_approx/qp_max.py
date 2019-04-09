"""
Solve a quadratic binary program with treatment_budget constraint.
For taking argmax of quadratic approximation to Q function.


ref: http://www.gurobi.com/documentation/7.0/examples/dense_py.html
"""
import numpy as np
from scipy import sparse
import pdb
try:
  from gurobipy import *
  GUROBI = True
except ImportError:
  import miosqp
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
    return qp_max_miosqp(M, r, budget)


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
  pdb.set_trace()
  return np.array([v.X for v in vars])


def qp_max_miosqp(M, r, budget):
  """
  miosqp notation:
    minimize        0.5 x' P x + q' x

    subject to      l <= A x <= u
                    x[i] in Z for i in i_idx
                    i_l[i] <= x[i] <= i_u[i] for i in i_idx

  :param M:
  :param r:
  :param budget:
  :return:
  """
  # Optimization problem definition
  L = M.shape[0]
  P = sparse.csc_matrix(2*M)
  q = np.zeros(L)
  A = sparse.csc_matrix(np.ones((1, L)))
  l = np.array([budget])
  u = np.array([budget])
  i_l = np.array([0 for _ in range(L)])
  i_u = np.array([1 for _ in range(L)])
  i_idx = np.array([i for i in range(L)])

  # Optimizer settings
  # Currently using settings from
  # https://github.com/oxfordcontrol/miosqp/blob/master/examples/random_miqp/run_example.py

  miosqp_settings = {
    # integer feasibility tolerance
    'eps_int_feas': 1e-03,
    # maximum number of iterations
    'max_iter_bb': 1000,
    # tree exploration rule
    #   [0] depth first
    #   [1] two-phase: depth first until first incumbent and then  best bound
    'tree_explor_rule': 1,
    # branching rule
    #   [0] max fractional part
    'branching_rule': 0,
    'verbose': False,
    'print_interval': 1}

  osqp_settings = {'eps_abs': 1e-03,
                   'eps_rel': 1e-03,
                   'eps_prim_inf': 1e-04,
                   'verbose': False}

  model = miosqp.MIOSQP()
  model.setup(P, q, A, l, u, i_idx, i_l, i_u, miosqp_settings, osqp_settings)
  results = model.solve()
  return results.x


