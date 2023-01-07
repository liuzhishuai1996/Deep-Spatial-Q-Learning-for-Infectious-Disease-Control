#
def environment_factory(environment_name, **kwargs):
  """
  :param feature_function:
  :param **kwargs: environment-specific keyword arguments
  :return: SpatialDisease environment
  """
  if environment_name == 'sis':
    from .sis import SIS
    return SIS(**kwargs)
  elif environment_name == 'Ebola':
    from .Ebola import Ebola
    return Ebola(**kwargs)
  elif environment_name == 'ContinuousGrav':
    from .ContinuousGrav import ContinuousGrav
    return ContinuousGrav(**kwargs)

