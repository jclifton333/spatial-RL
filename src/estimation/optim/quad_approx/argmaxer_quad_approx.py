import numpy as np
from .fit_quad_approx import get_quadratic_program_from_q
from .qp_max import qp_max
import time


def argmaxer_quad_approx(q, evaluation_budget, treatment_budget, env, ixs=None):
  """
  Take the argmax by fitting quadratic approximation to q and solving resulting binary quadratic program.
  """
  if ixs:
    L = len(ixs)
    treatment_budget = int(np.ceil((treatment_budget/env.L)*L))
  else:
    L = env.L
  M, r = get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env, L)
  a = qp_max(M, r, treatment_budget)
  return a