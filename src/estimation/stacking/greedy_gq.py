# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse
ggq implementation, *specifically* for stacking q functions.

See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf
     https://arxiv.org/pdf/1406.0764.pdf (Ertefaie, algo on pdf pg 16)

"q_features" and "X" corresponds to \varphi in GGQ paper (this is confusing and I will fix it, time permitting.)
"""
import numpy as np
from .compute_sample_bellman_error import compute_temporal_differences
import pdb


def q_features(data_block, q_list, intercept):
  q_list = np.array([q(data_block) for q in q_list])
  if intercept:
    q_list = np.vstack((np.ones(q_list.shape[1]), q_list))
  return q_list.T


def ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer,
               weights=None, intercept=True):

  q_of_X = np.dot(X, theta)

  def q_features_at_block(data_block):
      return q_features(data_block, q_list, intercept)

  def stacked_q_fn(data_block):
    return np.dot(q_features_at_block(data_block), theta)

  _, TD_times_X, X_hat = compute_temporal_differences(stacked_q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                      argmaxer, bootstrap_correction_weights=weights, q_of_X=q_of_X)
  return TD_times_X, X_hat


def ggq_pieces_repeated(theta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer,
                        weights_array, intercept=True):
  """
  Returns ggq_pieces weighted by each set weights in weights_list; for re-weighting according to bootstrap
  correction weights.
  :param theta:
  :param q1_list:
  :param q2_list:
  :param gamma:
  :param env:
  :param evaluation_budget:
  :param treatment_budget:
  :param q_features:
  :param argmaxer:
  :param weights_array: (B x T x L) array of weights
  :param intercept:
  :return:
  """
  TD_times_X, X_hat = [], []
  for q1_b, q2_b, weights_b in zip(q1_list, q2_list, weights_array):
    TD_times_X_b, X_hat_b = ggq_pieces(theta, [q1_b, q2_b], gamma, env, evaluation_budget, treatment_budget, X,
                                       argmaxer, weights_b, intercept=intercept)
    TD_times_X.append(TD_times_X_b)
    X_hat_b.append(X_hat_b)
  return np.vstack(TD_times_X), np.vstack(X_hat)


def update_theta(theta, alpha, gamma, TD_times_X, Xw, X_hat):
  Xw_times_Xhat = np.multiply(Xw.reshape(len(Xw), 1), X_hat)
  gradient_block = TD_times_X - gamma*Xw_times_Xhat
  theta += alpha * np.mean(gradient_block, axis=0)
  return theta


def update_w(w, beta, TD_times_X, X, Xw):
  Xw_times_X = np.multiply(Xw.reshape(len(Xw), 1), X)
  gradient_block = TD_times_X - Xw_times_X
  w += beta * np.mean(gradient_block, axis=0)
  return w


def update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat, project):
  Xw = np.dot(X, w)
  theta = update_theta(theta, alpha, gamma, TD_times_X, Xw, X_hat)
  if project:
    theta = np.max((theta, np.zeros(len(theta))), axis=0)
  w = update_w(w, beta, TD_times_X, X, Xw)
  return theta, w


def ggq_step(theta, w, alpha, beta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer, ixs,
             intercept, project):
  TD_times_X, X_hat = ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer, ixs,
                                 intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat, project)
  return theta, w


def ggq_step_repeated(theta, w, alpha, beta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, X,
                      argmaxer, weights_array, project=True, intercept=True):
  """
  GGQ step for repeated bootstrap samples.
  :param theta:
  :param w:
  :param alpha:
  :param beta:
  :param q_list:
  :param gamma:
  :param env:
  :param evaluation_budget:
  :param treatment_budget:
  :param X:
  :param argmaxer:
  :param ixs:
  :param intercept:
  :param project:
  :param weight_array:
  :return:
  """
  TD_times_X, X_hat = ggq_pieces_repeated(theta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, X,
                                          argmaxer, weights_array, intercept=intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat, project)
  return theta, w


def ggq(q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, argmaxer,
        bootstrap_weight_correction_arr=None, intercept=True, project=False):
  """
  :param project: Boolean for projecting theta onto positive R^n (for stacking, since weights should be positive.)
  """
  N_IT = 20
  NU = 1/25
  
  # n_features = len(q_list) + intercept
  n_features = 3  # Currently just combining 2 q functions (plus intercept)
  B = len(q1_list)  # Number of bootstrap replicates
  theta, w = np.zeros(n_features), np.zeros(n_features)
  X = np.zeros((0, n_features))
  for t in range(env.T - 1):
    data_block = env.X[t]
    for b in range(B):
      q1_b = q1_list[b]
      q2_b = q2_list[b]
      q_list = [q1_b, q2_b]
      q_hat = q_features(data_block, q_list, intercept)
    X = np.vstack((X, q_hat))

  for it in range(N_IT):
    alpha = NU / ((it + 1) * np.log(it + 2))
    beta = NU / (it + 1)
    theta, w = ggq_step_repeated(theta, w, alpha, beta, q1_list, q2_list, gamma, env, evaluation_budget,
                                 treatment_budget, X, argmaxer, weights_array=bootstrap_weight_correction_arr,
                                 intercept=intercept, project=project)
  return theta
