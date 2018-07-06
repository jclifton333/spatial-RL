import src.policies.reference_policies as ref
import src.policies.rollout_policies as roll


def policy_factory(policy_type):
  """
  :param policy_type: String in ['random', 'no_action', 'true_probs', 'rollout', 'network rollout',
  'one_step'].
  :return: Corresponding policy function.
  """
  if policy_type == 'random':
    return ref.random
  elif policy_type == 'no_action':
    return ref.no_action
  elif policy_type == 'true_probs':
    return ref.true_probs
  elif policy_type == 'true_probs_myopic':
    return ref.true_probs_myopic
  elif policy_type == 'rollout':
    return roll.rollout_policy
  elif policy_type == 'network rollout':
    return roll.network_features_rollout_policy
  elif policy_type == 'one_step':
    return roll.one_step_policy
  elif policy_type == 'treat_all':
    return ref.treat_all
  else:
    raise ValueError('Argument does not match any policy.')