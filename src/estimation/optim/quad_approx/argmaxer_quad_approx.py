from .fit_quad_approx import get_quadratic_program_from_q
from .qp_max import qp_max


def argmaxer_quad_approx(q, evaluation_budget, treatment_budget, env):
  """
  Take the argmax by fitting quadratic approximation to q and solving resulting binary quadratic program.
  """
  M, r = get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env)
  a = qp_max(M, r, treatment_budget)
  return a