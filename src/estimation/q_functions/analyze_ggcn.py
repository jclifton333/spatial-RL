import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, 'data')
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import pdb
import copy
import numpy as np
import pickle as pkl
from src.environments.Ebola import Ebola
from src.environments.sis import SIS
from src.environments.generate_network import lattice
import argparse
from src.estimation.q_functions.embedding import oracle_tune_ggcn
from sklearn.linear_model import LogisticRegression
from src.estimation.q_functions.model_fitters import SKLogit2
import statsmodels.api as sm
import pandas as pd
from src.utils.misc import kl


def load_saved_ggcn_data(fname):
  # Get data from file
  fname = os.path.join(data_dir, fname)
  data_dict = pkl.load(open(fname, 'rb'))

  if data_dict['settings']['env'] == 'Ebola':
    env = Ebola(learn_embedding=True)
  elif data_dict['settings']['env'] == 'sis':  # Assuming lattice
    L = data_dict['settings']['L']
    env = SIS(L, 0.0, lattice, learn_embedding=True)

  return data_dict, env


def oracle_tune_data_dict(data_dict, env, num_settings_to_try=5, verbose=True, n_epochs=50):
  X_raw_list, X_list, y_list, adjacency_list, eval_actions, true_probs = data_dict['X_raw_list'], data_dict['X_list'], \
                                                                         data_dict['y_list'], \
                                                                         data_dict['adjacency_list'], \
                                                                         data_dict['eval_actions'], \
                                                                         data_dict['true_probs']
  
  predictor, results = oracle_tune_ggcn(X_list, y_list, adjacency_list, env, eval_actions, true_probs,
                                        X_eval=X_list[-1], n_epoch=n_epochs,
                                        num_settings_to_try=num_settings_to_try, verbose=verbose)

  return predictor


def log_likelihood(lm_fitted, Z, w):
  phat = lm_fitted.predict_proba(Z)[:, 1]
  ll = kl(w, phat)
  return ll


def fit_and_compare_models(data_dict, env, num_settings_to_try=5, n_epochs=50):
  nn = oracle_tune_data_dict(data_dict, env, num_settings_to_try=num_settings_to_try, n_epochs=n_epochs)

  # Linear baseline
  X_raw_list, X_list, y_list = data_dict['X_raw_list'], data_dict['X_list'], data_dict['y_list']
  lm = LogisticRegression(C=1)
  lm_raw = LogisticRegression()
  X_raw = np.vstack(X_raw_list)
  X_ = np.vstack(X_list)
  y_ = np.hstack(y_list)
  lm.fit(X_, y_)
  lm_raw.fit(X_raw, y_)

  # Define q functions
  def q_nn(a):
    if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
      x = copy.copy(X_list[-1])
      x[:, 1] = a
    else:
      raise NotImplementedError
    return nn(x)

  def q_lm_raw(a):
    X_raw_ = copy.copy(X_raw_list[-1])
    X_raw_[:, 1] = a
    return lm_raw.predict_proba(X_raw_)[:, 1]

  # TODO: WIP! have to get data_block_from_action
  def q_lm(a):
    if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
      x = copy.copy(X_list[-1])
      x[:, 1] = a
    else:
      raise NotImplementedError
    return lm.predict_proba(x)[:, 1]

  # Evaluate
  eval_actions = data_dict['eval_actions']
  true_probs = data_dict['true_probs']
  lm_raw_probs = np.zeros(0)
  lm_probs = np.zeros(0)
  nn_probs = np.zeros(0)
  for a_ in eval_actions:
    nn_probs = np.hstack((nn_probs, q_nn(a_)))
    lm_raw_probs = np.hstack((lm_raw_probs, q_lm_raw(a_)))
    lm_probs = np.hstack((lm_probs, q_lm(a_)))
  # lm_score = kl(lm_probs, true_probs) 
  # nn_score = kl(nn_probs, true_probs)
  lm_score = np.mean(np.abs(lm_probs - true_probs))
  lm_raw_score = np.mean(np.abs(lm_raw_probs - true_probs)) 
  nn_score = np.mean(np.abs(nn_probs - true_probs))
  ll = log_likelihood(lm, X_, y_)
  ll_raw = log_likelihood(lm_raw, X_raw, y_)
  ll_nn = kl(y_, np.hstack([nn(x_) for x_ in X_list]))
  print(ll_raw, ll_nn)
  lm_predict_raw = lm_raw.predict(X_raw)
  lm_acc_raw = np.mean(lm_predict_raw == y_)

  print(f'nn score: {nn_score} lm raw score: {lm_raw_score} lm score: {lm_score}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument('--fname', type=str, default='data_t=8.p')
  parser.add_argument('--fname', type=str, default='Ebola_data_t=8.p')
  args = parser.parse_args()

  data_dict, env = load_saved_ggcn_data(args.fname)
  fit_and_compare_models(data_dict, env, num_settings_to_try=1, n_epochs=300)
