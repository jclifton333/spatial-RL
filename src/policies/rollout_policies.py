from src.estimation.q_functions.rollout import rollout
from src.estimation.q_functions.regressor import AutoRegressor
import numpy as np


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer']
  clf = classifier()
  target = np.hstack(env.y).astype(float)
  clf.fit(np.vstack(env.X), target)
  true_expected_counts = np.hstack(env.true_infection_probs)
  phat = clf.predict_proba(np.vstack(env.X))[:,-1]
  r2 = 1 - (
      np.sum((phat - true_expected_counts) ** 2) / np.sum((true_expected_counts - np.mean(true_expected_counts)) ** 2))
  print('R2: {}'.format(r2))

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a))[:,-1]

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer']
    auto_regressor = AutoRegressor(classifier, regressor)

    a = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer)
  return a


def network_features_rollout_policy(**kwargs):
  env, evaluation_budget, treatment_budget, regressor = \
    kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['regressor']
  argmax_actions, _ = network_features_rollout(env, evaluation_budget, treatment_budget, regressor())
  a = argmax_actions[-1]
  return a
