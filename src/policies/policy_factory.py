import src.policies.reference_policies as ref
import src.policies.q_function_policies as roll
import src.policies.policy_search as ps


def policy_factory(policy_type):
  """
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
  elif policy_type == 'fqi':
    return roll.rollout_policy
  elif policy_type == 'network fqi':
    return roll.network_features_rollout_policy
  elif policy_type == 'one_step':
    return roll.one_step_policy
  elif policy_type == 'treat_all':
    return ref.treat_all
  elif policy_type == 'dummy_stacked':
    return roll.dummy_stacked_q_policy
  elif policy_type == 'sis_stacked':
    return roll.sis_stacked_q_policy
  elif policy_type == 'SIS_model_based':
    return roll.SIS_model_based_policy
  elif policy_type == 'sis_model_based_one_step':
    return roll.sis_model_based_one_step
  elif policy_type == 'sis_one_step_stacked_q':
    return roll.sis_one_step_stacked_q_policy
  elif policy_type == 'sis_one_step_be_averaged':
    return roll.sis_one_step_be_averaged_policy
  elif policy_type == 'one_step_mse_averaged':
    return roll.one_step_mse_averaged
  elif policy_type == 'sis_two_step_mse_averaged':
    return roll.sis_two_step_mse_averaged
  elif policy_type == 'sis_mb_fqi':
    return roll.sis_mb_fqi
  elif policy_type == 'ebola_model_based_one_step':
    return roll.ebola_model_based_one_step
  elif policy_type == 'ebola_model_based_myopic':
    return roll.ebola_model_based_myopic
  elif policy_type == 'policy_search':
    return ps.policy_search_policy
  else:
    raise ValueError('Argument does not match any policy.')
