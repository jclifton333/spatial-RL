import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, 'data')
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import numpy as np
import pickle as pkl
from src.environments.Ebola import Ebola
import argparse
from src.estimation.q_functions.embedding import oracle_tune_ggcn
import statsmodels.api as sm
import pandas as pd



def load_saved_ebola_ggcn_data(fname):
  # Get data from file
  fname = os.path.join(data_dir, fname)
  data_dict = pkl.load(open(fname, 'rb'))

  # Construct env
  env = Ebola(learn_embedding=True)

  return data_dict, env


def oracle_tune_data_dict(data_dict, env, num_settings_to_try=5):
  X_list, y_list, adjacency_list, eval_actions, true_probs = data_dict['X_list'], data_dict['y_list'], \
    data_dict['adjacency_list'], data_dict['eval_actions'], data_dict['true_probs']

  predictor, results = oracle_tune_ggcn(X_list, y_list, adjacency_list, env, eval_actions, true_probs,
                                        num_settings_to_try=num_settings_to_try)
  results = pd.DataFrame.from_dict(results)
  y = results.score
  Z = results.loc[:, ~results.columns.isin(['score', 'neighbor_subset'])]
  return predictor, y, Z


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument('--fname', type=str, default='data_t=8.p')
  parser.add_argument('--fname', type=str, default='data_t=36.p')
  args = parser.parse_args()

  data_dict, env = load_saved_ebola_ggcn_data(args.fname)
  predictor, y, Z = oracle_tune_data_dict(data_dict, env, num_settings_to_try=10)
  mod = sm.OLS(y, Z)
  res = mod.fit()
  print(res.summary())
