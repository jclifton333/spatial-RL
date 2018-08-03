from src.environments.SIS import SIS
from src.environments.sis_infection_probs import infection_probability
from src.estimation.q_functions.rollout import rollout
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q
from src.estimation.stacking.greedy_gq import ggq
from src.estimation.stacking.stack_fitted_qs import compute_bootstrap_weight_correction
from src.estimation.model_based.SIS.fit import fit_transition_model
from src.estimation.model_based.SIS.simulate import simulate_from_SIS
from src.estimation.model_based.SIS.estimate_mb_q_fn import estimate_SIS_q_fn
from src.estimation.model_based.SIS.fit import fit_transition_model
from src.utils.misc import random_argsort
import numpy as np
import keras.backend as K
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
  features = np.vstack(env.X)

  if clf.condition_on_infection:
    X_raw = np.vstack(env.X_raw)
    clf_kwargs = {'infected_locations': np.where(X_raw[:, -1] == 1),
                  'not_infected_locations': np.where(X_raw[:, -1] == 0)}
    predict_proba_kwargs = {'infected_locations': np.where(env.X_raw[-1][:, -1] == 1),
                            'not_infected_locations': np.where(env.X_raw[-1][:, -1] == 0)}
  else:
    clf_kwargs = {}
    predict_proba_kwargs = {}

  clf.fit(features, target, weights, **clf_kwargs)
  true_expected_counts = np.hstack(env.true_infection_probs)
  phat = clf.predict_proba(features, **clf_kwargs)[:, -1]
  loss = np.mean((phat - true_expected_counts)**2)
  print('loss {} mean infection {} mean state {}'.format(loss,
                                                         np.mean(env.current_infected),
                                                         np.mean(env.current_state < 0)))

  def qfn(a):
    return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)[:, -1]

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def rollout_policy(**kwargs):
  if kwargs['rollout_depth'] == 0:
    a, q_model = one_step_policy(**kwargs)
  else:
    classifier, regressor, env, evaluation_budget, gamma, rollout_depth, treatment_budget, argmaxer = \
      kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], \
      kwargs['gamma'], kwargs['rollout_depth'], kwargs['treatment_budget'], kwargs['argmaxer']

    auto_regressor = AutoRegressor(classifier, regressor)

    q_model = rollout(rollout_depth, gamma, env, evaluation_budget, treatment_budget, auto_regressor, argmaxer)
    q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=q_model)
    a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, q_model


def SIS_model_based_policy(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, bootstrap, \
    rollout_depth, gamma, classifier, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['bootstrap'], \
    kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']

  auto_regressor = AutoRegressor(classifier, regressor)
  new_q_model = estimate_SIS_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth,
                                  treatment_budget, evaluation_budget, argmaxer, bootstrap)

  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=new_q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, new_q_model


def SIS_model_based_one_step(**kwargs):
  env, bootstrap, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['bootstrap'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  eta = fit_transition_model(env, bootstrap=bootstrap)
  one_step_q = partial(infection_probability, y=env.current_infected, s=env.current_state, eta=eta,
                       omega=0, L=env.L, adjacency_lists=env.adjacency_list)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  return a, None


def dummy_stacked_q_policy(**kwargs):
  """
  This is for testing stacking.
  """
  gamma, env, evaluation_budget, treatment_budget, argmaxer = kwargs['gamma'], kwargs['env'], \
    kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer']

  # Generate dummy q functions
  B = 2
  q1_list = []
  q2_list = []
  bootstrap_weights_list = []
  for b in range(B):
    q1_b = lambda x: np.random.random(x.shape[0])
    q2_b = lambda x: np.random.random(x.shape[0])
    bs_weights_b = np.random.exponential(size=(env.T, env.L))
    q1_list.append(q1_b)
    q2_list.append(q2_b)
    bootstrap_weights_list.append(bs_weights_b)

  bootstrap_weight_correction_arr = compute_bootstrap_weight_correction(bootstrap_weights_list)
  theta = ggq(q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, argmaxer,
              bootstrap_weight_correction_arr=bootstrap_weight_correction_arr)
  a = np.random.permutation(np.append(np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
  new_q_model = lambda x: theta[0]*q_0(x) + theta[1]*q_1(x)
  return a, new_q_model


def sis_one_step_stacked_q_policy(**kwargs):
  env = kwargs['env']

  # Generate bootstrap weights
  bootstrap_weights = np.random.exponential(size=(env.T, env.L))

  # Get model-based one-step q fn
  eta = fit_transition_model(env, bootstrap_weights=bootstrap_weights)
  q_mb =


def sis_stacked_q_policy(**kwargs):
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
