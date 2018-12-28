"""
Fit Q-function on (large) batch data and roll out without updating.
"""
# So stupid relative imports work
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)
import pdb

from src.environments.environment_factory import environment_factory
from src.environments.generate_network import lattice
from src.estimation.q_functions.one_step import fit_one_step_predictor
from src.estimation.q_functions.model_fitters import SKLogit2
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor


def generate_two_step_sis_data(L, time_horizon, number_of_data_points=1e5):
  # Check if data for these settings has already been generated
  data_dir = os.path.join(this_dir, 'data_for_prefit_policies')
  already_generated = False
  for fname in os.listdir(data_dir):
    if 'two-step-sis' in fname:
      if 'time_horizon={}'.format(time_horizon) in fname and 'L={}'.format(L) in fname:
        already_generated = True

  if not already_generated:
    number_of_episodes = int(np.floor(number_of_data_points / (L * time_horizon)))

    X_first_order = []
    X_second_order = []
    X_raw = []
    y = []
    for ep in range(number_of_episodes):
      print('episode {}'.format(ep))
      treatment_budget = int(np.floor(0.05 * L))
      env = environment_factory('sis', **{'L': L, 'omega': 0.0, 'generate_network': lattice})
      env.reset()

      # Initial steps
      dummy_action = np.concatenate((np.zeros(L - treatment_budget), np.ones(treatment_budget)))
      env.step(np.random.permutation(dummy_action))
      env.step(np.random.permutation(dummy_action))
      for t in range(time_horizon - 2):
        env.step(np.random.permutation(dummy_action))
      y += env.y
      X_raw += env.X_raw
      X_first_order += env.X
      X_second_order += env.X_2

    data = {'X_first_order': X_first_order, 'X_second_order': X_second_order, 'y': y, 'X_raw': X_raw}
    fname = os.path.join(data_dir, 'two-step-sis-time_horizon={}-L={}.p'.format(time_horizon, L))
    pkl.dump(data, open(fname, 'wb'))

  return


def two_step_sis_prefit(**kwargs):
  env, evaluation_budget, treatment_budget, argmaxer, bootstrap, q_fn, time_horizon = \
    kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], kwargs['bootstrap'], \
    kwargs['q_fn'], kwargs['planning_depth']

  if q_fn is None:  # Haven't fit yet
    # Load pre-saved data
    path_to_saved_data = \
      os.path.join(this_dir, 'data_for_prefit_policies/two-step-sis-time_horizon={}-L={}.p'.format(time_horizon, env.L))
    data = pkl.load(open(path_to_saved_data, 'rb'))
    X_raw, X, X_2, y = data['X_raw'], data['X_first_order'], data['X_second_order'], data['y']

    infected_locations = np.where(np.vstack(X_raw)[:, -1] == 1)
    not_infected_locations = np.where(np.vstack(X_raw)[:, -1] == 0)

    # One step
    clf = SKLogit2()
    clf.fit(np.vstack(X), np.hstack(y), None, False, infected_locations, not_infected_locations)

    def qfn_at_block(block_index, a):
      infected_locations_ = np.where(X_raw[block_index][:, -1] == 1)
      not_infected_locations_ = np.where(X_raw[block_index][:, -1] == 0)
      X_raw_ix = X_raw[block_index]
      X_raw_at_action = np.column_stack((X_raw_ix[:, 0], a, X_raw_ix[:, 2]))
      data_block_at_action = env.psi(X_raw_at_action, neighbor_order=1)
      return clf.predict_proba(data_block_at_action, infected_locations_, not_infected_locations_)

    # Back up once
    backup = []
    T = len(X)
    for t in range(T):
      # ToDo: make sure indexing of Xs and ys match up
      qfn_at_block_t = lambda a: qfn_at_block(t, a)
      a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
      q_max = qfn_at_block_t(a_max)
      backup_at_t = y[t] + q_max
      backup.append(backup_at_t)

    # Fit backup-up q function
    reg = RandomForestRegressor(n_estimators=200)
    reg.fit(np.vstack(X_2), np.hstack(backup))

    def q_fn(a, env):
      return reg.predict(env.data_block_at_action(-1, a, neighbor_order=2))

  a = argmaxer(lambda a: q_fn(a, env), evaluation_budget, treatment_budget, env)
  return a, {'q_fn': q_fn}

