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
import argparse
from src.estimation.q_functions.embedding import oracle_tune_ggcn
from sklearn.linear_model import LogisticRegression
from src.estimation.q_functions.model_fitters import SKLogit2
import statsmodels.api as sm
import pandas as pd
from src.utils.misc import kl


def load_saved_ebola_ggcn_data(fname):
  # Get data from file
  fname = os.path.join(data_dir, fname)
  data_dict = pkl.load(open(fname, 'rb'))

  # Construct env
  env = Ebola(learn_embedding=True)

  return data_dict, env


def oracle_tune_data_dict(data_dict, env, num_settings_to_try=5, verbose=True):
  X_list, y_list, adjacency_list, eval_actions, true_probs = data_dict['X_list'], data_dict['y_list'], \
    data_dict['adjacency_list'], data_dict['eval_actions'], data_dict['true_probs']

  predictor, results = oracle_tune_ggcn(X_list, y_list, adjacency_list, env, eval_actions, true_probs,
                                        X_eval=X_list[-1],
                                        num_settings_to_try=num_settings_to_try, verbose=verbose)

  return predictor


def fit_and_compare_models(data_dict, env, num_settings_to_try=5):
  nn = oracle_tune_data_dict(data_dict, env, num_settings_to_try=num_settings_to_try)

  # Linear baseline
  X_list, y_list = data_dict['X_list'], data_dict['y_list']
  lm = LogisticRegression()
  lm.fit(np.vstack(X_list), np.hstack(y_list))

  # Define q functions
  def q_nn(a):
    X_raw_ = copy.copy(X_list[-1])
    X_raw_[:, 1] = a
    return nn(X_raw_)

  def q_lm(a):
    X_raw_ = copy.copy(X_list[-1])
    X_raw_[:, 1] = a
    return lm.predict_proba(X_raw_)[:, 1]

  # Evaluate
  lm_score = 0.
  nn_score = 0.
  eval_actions = data_dict['eval_actions']
  true_probs = data_dict['true_probs']
  lm_probs = np.zeros(0)
  nn_probs = np.zeros(0)
  for a_ in eval_actions:
    nn_probs = np.hstack((nn_probs, q_nn(a_)))
    lm_probs = np.hstack((lm_probs, q_lm(a_)))
  # lm_score = kl(lm_probs, true_probs) 
  # nn_score = kl(nn_probs, true_probs) 
  lm_score = np.mean(np.abs(lm_probs - true_probs)) 
  nn_score = np.mean(np.abs(nn_probs - true_probs))

  print(f'nn score: {nn_score} lm score: {lm_score}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument('--fname', type=str, default='data_t=8.p')
  parser.add_argument('--fname', type=str, default='data_t=36.p')
  args = parser.parse_args()

  data_dict, env = load_saved_ebola_ggcn_data(args.fname)
  fit_and_compare_models(data_dict, env, num_settings_to_try=5)
