try:
  from .quad_approx.argmaxer_quad_approx import argmaxer_quad_approx, argmaxer_sequential_quad_approx
except ImportError:
  print('Gurobi not available.  Can\'t use quad_approx.')

from .sweep.argmaxer_sweep import argmaxer_sweep
import numpy as np
import logging
import pdb
from scipy.misc import comb
from itertools import combinations


def argmaxer_random(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  # Placeholder argmax function for debugging.
  if ixs is not None:
    L = len(ixs)
    treatment_budget = int(np.ceil((treatment_budget / env.L) * L))
  else:
    L = env.L
  dummy = np.append(np.ones(treatment_budget), np.zeros(L - treatment_budget))
  return np.random.permutation(dummy)


def argmaxer_global(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  HARD_EVALUATION_LIMIT = 1000
  assert(comb(env.L, treatment_budget) < HARD_EVALUATION_LIMIT,
         '(L choose treatment_budget) greater than HARD_EVALUATION_LIMIT.')
  all_ix_combos = combinations(range(env.L), treatment_budget)
  q_best = float('inf')
  a_best = None
  for ixs in all_ix_combos:
    a = np.zeros(env.L)
    a[ixs] = 1
    q_sum = np.sum(q_fn(a))
    if q_sum < q_best:
      q_best = q_sum
      a_best = a
  return a_best


def argmaxer_factory(choice):
  """
  :param choice: str in ['sweep', 'quad_approx']
  :return:
  """
  if choice == 'sweep':
    return argmaxer_sweep
  elif choice == 'quad_approx':
    return argmaxer_quad_approx
  elif choice == 'sequential_quad_approx':
    return argmaxer_sequential_quad_approx
  elif choice == 'random':
    return argmaxer_random
  elif choice == 'global':
    logging.warning('Using global argmaxer; this may be especially slow.')
    return argmaxer_global
  else:
    raise ValueError('Argument is not a valid argmaxer name.')
