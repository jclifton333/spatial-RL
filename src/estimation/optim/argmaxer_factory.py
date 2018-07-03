from .quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
from .sweep.argmaxer_sweep import argmaxer_sweep


def argmaxer_factory(choice):
  """
  :param choice: str in ['sweep', 'quad_approx']
  :return:
  """
  if choice == 'sweep':
    return argmaxer_sweep
  elif choice == 'quad_approx':
    return argmaxer_quad_approx
  else:
    raise ValueError('Argument is not a valid argmaxer name.')
