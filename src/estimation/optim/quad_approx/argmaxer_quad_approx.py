import numpy as np
from .fit_quad_approx import get_quadratic_program_from_q
from .qp_max import qp_max
import time


def argmaxer_quad_approx(q, evaluation_budget, treatment_budget, env, initial=None, ixs=None):
  """
  Take the argmax by fitting quadratic approximation to q and solving resulting binary quadratic program.
  """
  if ixs is not None:
    treatment_budget = int(np.ceil((treatment_budget/env.L)*len(ixs)))
  M, r = get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env, ixs, initial=None)
  a = qp_max(M, r, treatment_budget)
  return a


def argmaxer_sequential_quad_approx(q, evaluation_budget, treatment_budget, env, sequence_length=3, ixs=None):
  """

  :param q:
  :param evaluation_budget:
  :param treatment_budget:
  :param env:
  :param sequece_length:
  :param ixs:
  :return:
  """
  # Get initial action
  initial = np.zeros(env.L)
  q_hat = q(initial)
  treat_ixs = np.argsort(-q_hat, treatment_budget)
  initial[treat_ixs] = 1

  for i in range(sequence_length):
    initial = argmaxer_quad_approx(q, evaluation_budget, treatment_budget, env, initial=initial)
  a = initial
  return a


