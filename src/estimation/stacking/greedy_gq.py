# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse
ggq implementation, *specifically* for stacking q functions.

See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf
     https://arxiv.org/pdf/1406.0764.pdf (Ertefaie, algo on pdf pg 16)

phi is the feature function in the GGQ paper; here, the features are each q-function that's being stacked evaluated
at the original features (which may themselves be functions of the X_raw, the raw [S A Y] blocks...)
"""
import numpy as np
from .compute_sample_bellman_error import compute_temporal_differences
import pdb


def q_feature_function(data_block, q_list, intercept):
  q_list = np.array([q(data_block) for q in q_list])
  if intercept:
    q_list = np.vstack((np.ones(q_list.shape[1]), q_list))
  return q_list.T


def temporal_differences(stacked_q_fn, feature_function, gamma, env, evaluation_budget, treatment_budget, argmaxer,
                         bootstrap_correction_weight, q_of_phi):

  # Evaluate stacked q_fn at Xp1
  for data_block in env.X:
    phi_at_data_block = feature_function(data_block)
    stacked_q_fn_at_X =
    X_hat = argmaxer(stacked_q_fn, evaluation_budget, treatment_budget, env)


  return


def ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, phi, argmaxer,
               weights=None, intercept=True):

  q_of_phi = np.dot(phi, theta)

  def q_features_at_block(data_block):
      return phi(data_block, q_list, intercept)

  def stacked_q_fn(data_block):
    return np.dot(q_features_at_block(data_block), theta)

  _, TD_times_phi, X_hat = temporal_differences(stacked_q_fn, gamma, env, evaluation_budget, treatment_budget,
                                                argmaxer, bootstrap_correction_weights=weights, q_of_X=q_of_X)

  # X_hat are env features with actions as the argmax of stacked q_fn; we need to transform these into
  # q features
  phi_hat = np.vstack([q_features_at_block(X_hat_block) for X_hat_block in X_hat])

  return TD_times_q_features, phi_hat


def ggq_pieces_repeated(theta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, phi, argmaxer,
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
  :param phi:
  :param argmaxer:
  :param weights_array: (B x T x L) array of weights
  :param intercept:
  :return:
  """
  TD_times_phi, phi_hat = [], []
  for q1_b, q2_b, weights_b in zip(q1_list, q2_list, weights_array):
    TD_times_phi_b, phi_hat_b = ggq_pieces(theta, [q1_b, q2_b], gamma, env, evaluation_budget, treatment_budget, phi,
                                           argmaxer, weights_b, intercept=intercept)
    TD_times_phi.append(TD_times_phi_b)
    phi_hat.append(phi_hat_b)
  return np.vstack(TD_times_phi), np.vstack(phi_hat)


def update_theta(theta, alpha, gamma, TD_times_phi, phi_dot_w, phi_hat):
  phi_dot_w_times_phi_hat = np.multiply(phi_dot_w.reshape(len(phi_dot_w), 1), phi_hat)
  gradient_block = TD_times_phi - gamma*phi_dot_w_times_phi_hat
  theta += alpha * np.mean(gradient_block, axis=0)
  return theta


def update_w(w, beta, TD_times_phi, phi, phi_dot_w):
  phi_dot_w_times_phi_hat = np.multiply(phi_dot_w.reshape(len(phi_dot_w), 1), phi)
  gradient_block = TD_times_phi - phi_dot_w_times_phi_hat
  w += beta * np.mean(gradient_block, axis=0)
  return w


def update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_phi, phi, phi_hat, project):
  phi_dot_w = np.dot(phi, w)
  theta = update_theta(theta, alpha, gamma, TD_times_phi, phi_dot_w, phi_hat)
  if project:
    theta = np.max((theta, np.zeros(len(theta))), axis=0)
  w = update_w(w, beta, TD_times_phi, phi, phi_dot_w)
  return theta, w


def ggq_step(theta, w, alpha, beta, q_list, gamma, env, evaluation_budget, treatment_budget, phi, argmaxer, ixs,
             intercept, project):
  TD_times_phi, phi_hat = ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, phi, argmaxer, ixs,
                                     intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_phi, phi, phi_hat, project)
  return theta, w


def ggq_step_repeated(theta, w, alpha, beta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget, phi,
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
  :param phi:
  :param argmaxer:
  :param ixs:
  :param intercept:
  :param project:
  :param weight_array:
  :return:
  """
  TD_times_phi, phi_hat = ggq_pieces_repeated(theta, q1_list, q2_list, gamma, env, evaluation_budget, treatment_budget,
                                              phi, argmaxer, weights_array, intercept=intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_phi, phi, phi_hat, project)
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
  phi = np.zeros((0, n_features))
  for t in range(env.T - 1):
    data_block = env.X[t]
    for b in range(B):
      q1_b = q1_list[b]
      q2_b = q2_list[b]
      q_list = [q1_b, q2_b]
      phi_at_data_block = q_features(data_block, q_list, intercept)
    phi = np.vstack((phi, phi_at_data_block))

  for it in range(N_IT):
    alpha = NU / ((it + 1) * np.log(it + 2))
    beta = NU / (it + 1)
    theta, w = ggq_step_repeated(theta, w, alpha, beta, q1_list, q2_list, gamma, env, evaluation_budget,
                                 treatment_budget, phi, argmaxer, weights_array=bootstrap_weight_correction_arr,
                                 intercept=intercept, project=project)
  return theta
