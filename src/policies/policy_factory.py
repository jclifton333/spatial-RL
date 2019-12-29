import src.policies.reference_policies as ref
import src.policies.q_function_policies as roll
import src.policies.policy_search as ps
import src.policies.prefit_policies as prefit
import src.policies.model_selection_policies as model_selection
import src.policies.continuation_policies as continuation
import src.policies.dyna_policies as dyna
import src.policies.evaluation_policies as eval


def policy_factory(policy_type):
  """
  :return: Corresponding policy function.
  """
  if policy_type == 'random':
    return ref.random
  elif policy_type == 'treat_first':
    return ref.treat_first
  elif policy_type == 'sis_one_step_dyna_space_filling':
    return dyna.sis_one_step_dyna_space_filling
  elif policy_type == 'sis_aic_two_step':
    return model_selection.sis_aic_two_step
  elif policy_type == 'ebola_aic_one_step':
    return model_selection.ebola_aic_one_step
  elif policy_type == 'sis_one_step_continuation':
    return continuation.sis_one_step_continuation
  elif policy_type == 'sis_aic_one_step':
    return model_selection.sis_aic_one_step
  elif policy_type == 'ebola_aic_two_step':
    return model_selection.ebola_aic_two_step
  elif policy_type == 'sis_local_aic_one_step':
    return model_selection.sis_local_aic_one_step
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
  elif policy_type == 'two_step_stacked':
    return roll.two_step_stacked
  elif policy_type == 'two_step_higher_order':
    return roll.two_step_higher_order
  elif policy_type == 'treat_all':
    return ref.treat_all
  elif policy_type == 'sis_model_based_one_step':
    return roll.sis_model_based_one_step
  elif policy_type == 'one_step_mse_averaged':
    return roll.one_step_mse_averaged
  elif policy_type == 'sis_two_step_mse_averaged':
    return roll.sis_two_step_mse_averaged
  elif policy_type == 'two_step_mb':
    return roll.two_step_mb
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
  elif policy_type == 'one_step_truth_augmented':
    return roll.one_step_truth_augmented
  elif policy_type == 'one_step_projection_combo':
    return roll.one_step_projection_combo
  elif policy_type == 'two_step_random':
    return eval.two_step_random
  elif policy_type == 'two_step_mb_myopic':
    return eval.two_step_mb_myopic
  elif policy_type == 'two_step_mb_constant_cutoff':
    return eval.two_step_mb_constant_cutoff
  elif policy_type == 'two_step_mb_constant_cutoff_test':
    return eval.two_step_mb_constant_cutoff_test
  elif policy_type == 'one_step_eval':
    return eval.one_step_eval
  elif policy_type == 'one_step_bins':
    return eval.one_step_bins
  else:
    raise ValueError('Argument does not match any policy.')
