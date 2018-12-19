def environment_factory(environment_name, **kwargs):
  """
  :param environment_name: String in ['sis', 'Gravity']
  :param feature_function:
  :param **kwargs: environment-specific keyword arguments
  :return: SpatialDisease environment
  """


  VALID_ENVIRONMENT_NAMES = ['sis', 'Gravity']
  if environment_name == 'sis':
    from .sis import SIS
    return SIS(**kwargs)
  elif environment_name == 'Gravity':
    from .Ebola import Ebola
    return Ebola(**kwargs)
  else:
    raise ValueError('environment_name not in {}'.format(VALID_ENVIRONMENT_NAMES))
