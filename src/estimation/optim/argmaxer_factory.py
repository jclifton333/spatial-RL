try:
  from .quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
except ImportError:
  print('Gurobi not available.  Can\'t use quad_approx.')

from .sweep.argmaxer_sweep import argmaxer_sweep
import numpy as np


def argmaxer_random(q_fn, evaluation_budget, treatment_budget, env, ixs=None):
  # For when you just need a placeholder.
  if ixs:
    L = len(ixs)
    treatment_budget = int(np.ceil((treatment_budget / env.L) * L))
  else:
    L = env.L
  dummy = np.append(np.ones(treatment_budget), np.zeros(L - treatment_budget))
  return np.random.permutation(dummy)


def argmaxer_factory(choice):
  """
  :param choice: str in ['sweep', 'quad_approx']
  :return:
  """
  if choice == 'sweep':
    return argmaxer_sweep
  elif choice == 'quad_approx':
    return argmaxer_quad_approx
  elif choice == 'random':
    return argmaxer_random
  else:
    raise ValueError('Argument is not a valid argmaxer name.')
