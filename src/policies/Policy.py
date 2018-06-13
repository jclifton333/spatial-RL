from .referencePolicies import random, no_action, true_probs
from .rolloutPolicies import rollout_policy, network_level_rollout_policy


def policy_factory(policy_type):
  """
  :param policy_type: String in ['random', 'no_action', 'true_probs', 'rollout', 'network rollout'].
  :return: Corresponding policy function.
  """
  if policy_type == 'random':
    return random
  elif policy_type == 'no_action':
    return no_action
  elif policy_type == 'true_probs':
    return true_probs
  elif policy_type == 'rollout':
    return rollout_policy
  elif policy_type == 'network rollout':
    return network_level_rollout_policy
  else:
    raise ValueError('Argument does not match any policy.')
