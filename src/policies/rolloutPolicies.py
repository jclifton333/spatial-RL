from src.estimation.Fitted_Q import rollout, network_features_rollout
from src.estimation.AutoRegressor import AutoRegressor
from src.estimation.Q_functions import Q_max
import numpy as np


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  clf = classifier()
  clf.fit(np.vstack(env.X).reshape(1,-1), np.hstack(env.y).astype(float))
  Qfn = lambda X: clf.predict_proba(X)[:,-1]
  _, a, _ = Q_max(Qfn, evaluation_budget, treatment_budget, env.L)
  return a


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget']
    auto_regressor = AutoRegressor(classifier, regressor)

    argmax_actions, _, _, _, _ = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget,
                                         auto_regressor, [])
    a = argmax_actions[-1]
  return a


def network_features_rollout_policy(**kwargs):
  env, evaluation_budget, treatment_budget, regressor = \
    kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['regressor']
  argmax_actions, _ = network_features_rollout(env, evaluation_budget, treatment_budget, regressor())
  a = argmax_actions[-1]
  return a
