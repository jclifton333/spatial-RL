"""
Fit Q-function on (large) batch data and roll out without updating.
"""
import numpy as np
import pickle as pkl
from ..environments.environment_factory import environment_factory
from ..environments.generate_network import lattice
from ..estimation.q_functions.one_step import fit_one_step_predictor


def generate_two_step_sis_data():
  number_of_data_points = 1e6
  L = 100
  time_horizon = 50
  number_of_episodes = int(np.floor(number_of_data_points / (L * time_horizon)))

  X_first_order = []
  X_second_order = []
  y = []
  for _ in range(number_of_episodes):
    treatment_budget = int(np.floor(0.05 * L))
    env = environment_factory('sis', **{'L': L, 'omega': 0.0 , 'generate_network': lattice})
    env.reset()

    # Initial steps
    dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
    env.step(np.random.permutation(dummy_action))
    env.step(np.random.permutation(dummy_action))
    for t in range(time_horizon - 2):
      env.step(np.random.permutation(dummy_action))
    y.append(np.hstack(env.y))
    X_first_order.append(np.vstack(env.X))
    X_second_order.append(np.vstack(env.X_2))

  X_first_order = np.vstack(X_first_order)
  X_second_order = np.vstack(X_second_order)
  y = np.hstack(y)
  data = {'X_first_order': X_first_order, 'X_second_order': X_second_order, 'y': y}
  fname = './data_for_prefit_policies/two-step-sis.p'
  pkl.dump(data, open(fname, 'wb'))

  return


def two_step_sis_prefit(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, q_fn = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['q_fn']

  if q_fn is None:  # Haven't fit yet
    # One step
    clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights)
    def qfn_at_block(block_index, a):
      return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

    # Back up once
    backup = []
    for t in range(env.T-1):
      qfn_at_block_t = lambda a: qfn_at_block(t, a)
      a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
      q_max = qfn_at_block_t(a_max)
      backup_at_t = env.y[t] + q_max
      backup.append(backup_at_t)

    # Fit backup-up q function
    reg = regressor()
    reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

    def q_fn(a, env):
      return reg.predict(env.data_block_at_action(-1, a))

  a = argmaxer(lambda a: q_fn(a, env), evaluation_budget, treatment_budget, env)
  return a, {'q_fn': q_fn}
