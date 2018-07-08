# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse
ggq implementation
See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf
     https://arxiv.org/pdf/1406.0764.pdf (Ertefaie, algo on pdf pg 16)

My X corresponds to \varphi in GGQ paper.
"""
import numpy as np
from src.estimation.optim.q_max import q_max_all_states
import pdb


def q_features(data_block, q_list, intercept):
  q_list = np.array([q(data_block) for q in q_list])
  if intercept:
    q_list = np.append([1], q_list)
  return q_list


def ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer,
               intercept=True):
  assert env.T > 1

  # Evaluate Q
  q = np.dot(X, theta)
  
  # Get Qmax
  def q_features_at_block(data_block):
    return q_features(data_block, q_list, intercept)

  def stacked_q_fn(data_block):
    return np.dot(q_features_at_block(data_block), theta)

  q_max, argmax_list = q_max_all_states(env, evaluation_budget, treatment_budget, stacked_q_fn, argmaxer)
  q_max = q_max[1:,]

  # Compute TD * semi-gradient and q features at argmax (X_hat)
  TD = np.hstack(env.y[:-1]).astype(float) + gamma*q_max.flatten() - q
  TD = TD.reshape(TD.shape[0],1)
  TD_times_X = np.multiply(TD, X)
  X_hat = np.vstack([q_features_at_block(x) for x in argmax_list[1:]])
  
  return TD_times_X, X_hat


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


def ggq_step(theta, w, alpha, beta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer, intercept,
             project):
  TD_times_X, X_hat = ggq_pieces(theta, q_list, gamma, env, evaluation_budget, treatment_budget, X, argmaxer, intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat, project)
  return theta, w


def ggq(q_list, gamma, env, evaluation_budget, treatment_budget, argmaxer, intercept=True, project=False):
  """
  :param project: Boolean for projecting theta onto positive R^n (for stacking, since weights should be positive.)
  """
  N_IT = 20
  NU = 1/25
  
  n_features = len(q_list) + intercept
  theta, w = np.zeros(n_features), np.zeros(n_features)
  X = np.vstack([q_features(data_block, q_list, intercept) for data_block in env.X[:-1]])

  for it in range(N_IT):
    alpha = NU / ((it + 1) * np.log(it + 2))
    beta = NU / (it + 1)
    theta, w = ggq_step(theta, w, alpha, beta, q_list, gamma, env, evaluation_budget, treatment_budget, X,
                        argmaxer, intercept=intercept, project=project)
  return theta
  
  
  
  