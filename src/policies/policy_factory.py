import src.policies.reference_policies as ref
import src.policies.q_function_policies as roll
import src.policies.policy_search as ps
import src.policies.prefit_policies as prefit


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
  elif policy_type == 'one_step':
    return roll.one_step_policy
  elif policy_type == 'two_step':
    return roll.two_step
  elif policy_type == 'two_step_higher_order':
    return roll.two_step_higher_order
  elif policy_type == 'treat_all':
    return ref.treat_all
  elif policy_type == 'sis_stacked':
    return roll.sis_model_based_one_step
  elif policy_type == 'one_step_mse_averaged':
    return roll.one_step_mse_averaged
  elif policy_type == 'sis_two_step_mse_averaged':
    return roll.sis_two_step_mse_averaged
  elif policy_type == 'sis_two_step_mb':
    return roll.sis_two_step_mb
  elif policy_type == 'sis_mb_fqi':
    return roll.sis_mb_fqi
  elif policy_type == 'gravity_model_based_one_step':
    return roll.gravity_model_based_one_step
  elif policy_type == 'gravity_model_based_myopic':
    return roll.continuous_model_based_myopic
  elif policy_type == 'sis_model_based_myopic':
    return roll.sis_model_based_myopic
  elif policy_type == 'policy_search':
    return ps.policy_search_policy
  elif policy_type == 'sis_one_step_equal_averaged':
    return roll.sis_one_step_equal_averaged
  elif policy_type == 'one_step_stacked':
    return roll.one_step_stacked
  elif policy_type == 'two_step_sis_prefit':
    return prefit.two_step_sis_prefit
  else:
    raise ValueError('Argument does not match any policy.')
