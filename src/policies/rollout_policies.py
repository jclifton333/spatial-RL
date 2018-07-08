from src.estimation.q_functions.rollout import rollout
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q
from src.estimation.model_based.SIS.fit import fit_transition_model
from src.estimation.model_based.SIS.simulate import simulate_from_SIS
import numpy as np
import pdb
from functools import partial


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
  return a, None


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer, train_ixs = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer'], kwargs['train_ixs']

    auto_regressor = AutoRegressor(classifier, regressor)

    q_model = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer,
                      ixs=train_ixs)
    q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
    a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  return a, q_model


def SIS_model_based_policy(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, q_model, train_ixs = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['q_model'], kwargs['train_ixs']

  # Need to fit q_model if it hasn't been already
  if q_model is None:
    rollout_depth, gamma, classifier, regressor = \
      kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']
    auto_regressor = AutoRegressor(classifier, regressor)
    q_model = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer)

  eta = fit_transition_model(env, train_ixs)
  simulation_env = simulate_from_SIS(env, eta, planning_depth, q_model, argmaxer, evaluation_budget,
                                     treatment_budget, train_ixs)
  kwargs['env'] = simulation_env
  a, new_q_model = rollout_policy(**kwargs)
  return a, new_q_model


def SIS_stacked_q_policy(**kwargs):
  q_mf = rollout_policy(**kwargs)
  q_mb = SIS_model_based_policy(**kwargs)


# def network_features_rollout_policy(**kwargs):
#   env, evaluation_budget, treatment_budget, regressor = \
#     kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['regressor']
#   argmax_actions, _ = network_features_rollout(env, evaluation_budget, treatment_budget, regressor())
#   a = argmax_actions[-1]
#   return a
