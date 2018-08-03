from .SIS import SIS
# from .Ebola import Ebola


def environment_factory(environment_name, **kwargs):
  """
  :param environment_name: String in ['SIS', 'Ebola']
  :param feature_function:
  :param **kwargs: environment-specific keyword arguments
  :return: SpatialDisease environment
  """


  VALID_ENVIRONMENT_NAMES = ['SIS', 'Ebola']
  if environment_name == 'SIS':
    return SIS(**kwargs)
  elif environment_name == 'Ebola':
    return Ebola()
  else:
    raise ValueError('environment_name not in {}'.format(VALID_ENVIRONMENT_NAMES))
