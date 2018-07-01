"""
Solve a quadratic binary program with treatment_budget constraint.
For taking argmax of quadratic approximation to Q function.
"""
import gurobipy as gb
import numpy as np


def qp_max(M, b, r, constraint):
  """
  max x'Mx + x'b + r
  s.t. 1'x = constraint

  :param M:
  :param b:
  :param r:
  :param constraint:
  :return:
  """

  m = gb.Model('qip')
  L = len(b)
  # ToDo: add variables

  # Define objective
  obj = np.dot(np.dot(M, x), x) + np.dot(b, x) + r
  m.setObjective(obj)

  # Define constraint
  m.addConstr(np.dot(np.ones(b), x) == constraint)

  # Optimize
  m.optimize()