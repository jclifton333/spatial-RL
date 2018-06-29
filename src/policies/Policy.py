from .referencePolicies import random, no_action, true_probs, true_probs_myopic, treat_all
from .rolloutPolicies import rollout_policy, network_features_rollout_policy, one_step_policy


def policy_factory(policy_type):
  """
  :param policy_type: String in ['random', 'no_action', 'true_probs', 'rollout', 'network rollout',
  'one_step'].
  :return: Corresponding policy function.
  """
  if policy_type == 'random':
    return random
  elif policy_type == 'no_action':
    return no_action
  elif policy_type == 'true_probs':
    return true_probs
  elif policy_type == 'true_probs_myopic':
    return true_probs_myopic
  elif policy_type == 'rollout':
    return rollout_policy
  elif policy_type == 'network rollout':
    return network_features_rollout_policy
  elif policy_type == 'one_step':
    return one_step_policy
  elif policy_type == 'treat_all':
    return treat_all
  else:
    raise ValueError('Argument does not match any policy.')
