from .referencePolicies import random, no_action, true_probs
from .rolloutPolicies import rollout

def policy_factory(policy_type):
  """
  :param policy_type: String in ['Random', 'NoAction', 'TrueProbs', 'Rollout'].
  :return: Corresponding policy function.
  """
  if policy_type == 'Random':
    return random
  elif policy_type == 'NoAction':
    return no_action
  elif policy_type == 'TrueProbs':
    return true_probs
  elif policy_type == 'Rollout':
    return rollout
  else:
    raise ValueError('Argument does not match any policy.')
