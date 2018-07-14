from src.estimation.q_functions.rollout import rollout
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q
from src.estimation.stacking.greedy_gq import ggq
from src.estimation.model_based.SIS.fit import fit_transition_model
from src.estimation.model_based.SIS.simulate import simulate_from_SIS
from src.estimation.model_based.SIS.estimate_mb_q_fn import estimate_SIS_q_fn
import numpy as np
import pdb
from functools import partial


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  clf = classifier()
  target = np.hstack(env.y).astype(float)
  clf.fit(np.vstack(env.X), target, weights)
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
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, q_model, train_ixs, bootstrap, \
    rollout_depth, gamma, classifier, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['q_model'], kwargs['train_ixs'], kwargs['bootstrap'], \
    kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']

  new_q_model = estimate_SIS_q_fn(env, classifier, regressor, rollout_depth, gamma, planning_depth,
                                  q_model, treatment_budget, evaluation_budget, argmaxer, train_ixs,
                                  bootstrap)

  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=new_q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  return a, new_q_model


def dummy_stacked_q_policy(**kwargs):
  """
  This is for testing stacking.
  """
  q_0 = lambda x: np.zeros(x.shape[0])
  q_1 = lambda x: np.ones(x.shape[0])
  q_list = [q_0, q_1]
  gamma, env, evaluation_budget, treatment_budget, argmaxer = kwargs['gamma'], kwargs['env'], \
    kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer']
  ixs = [0, 1, 2]
  ixs_list = [ixs for _ in range(env.T)]
  theta = ggq(q_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, ixs_list)
  a = np.random.permutation(np.append(np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
  new_q_model = lambda x: theta[0]*q_0(x) + theta[1]*q_1(x)
  return a, new_q_model


def SIS_stacked_q_policy(**kwargs):
  env = kwargs['env']
  train_ixs, test_ixs = env.train_test_split()
  kwargs['train_ixs'] = train_ixs
  _, q_mf = rollout_policy(**kwargs)
  _, q_mb = SIS_model_based_policy(**kwargs)
  q_list = [q_mf, q_mb]
  gamma, evaluation_budget, treatment_budget, argmaxer = kwargs['gamma'], kwargs['evaluation_budget'], \
                                                         kwargs['treatment_budget'], kwargs['argmaxer']
  ixs = [0, 1, 2]
  ixs_list = [ixs for _ in range(env.T)]
  theta = ggq(q_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, ixs_list)
  q_model = lambda x: theta[0]*q_mf(x) + theta[1]*q_mb(x)
  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  return a, q_hat


# def network_features_rollout_policy(**kwargs):
#   env, evaluation_budget, treatment_budget, regressor = \
#     kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['regressor']
#   argmax_actions, _ = network_features_rollout(env, evaluation_budget, treatment_budget, regressor())
#   a = argmax_actions[-1]
#   return a
