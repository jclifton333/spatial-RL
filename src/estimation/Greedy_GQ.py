# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse

"""

import numpy as np
from .Q_functions import Q_max_all_states
from .Fitted_Q import rollout_Q_features
import pdb

'''
GGQ implementation
See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf
     https://arxiv.org/pdf/1406.0764.pdf (Ertefaie, algo on pdf pg 16)
'''

def GGQ_pieces(theta, rollout_Q_function_list, gamma, 
                 env, evaluation_budget, treatment_budget, X, intercept=True):   
  assert env.T > 1
  
  rollout_Q_features_at_block = lambda data_block: rollout_Q_features(data_block, rollout_Q_function_list, intercept)

  #Evaluate Q 
  Q = np.dot(X, theta)
  
  #Get Qmax  
  Q_fn = lambda data_block: np.dot(rollout_Q_features_at_block(data_block), theta)
  Qmax, Qargmax, _, _ = Q_max_all_states(env, evaluation_budget, treatment_budget, Q_fn)
  Qmax = Qmax[1:,]
  X_hat = np.vstack([rollout_Q_features_at_block(x) for x in Qargmax[1:]])
  
  #Compute TD * semi-gradient
  TD = np.hstack(env.y[:-1]).astype(float) + gamma*Qmax.flatten() - Q
  TD = TD.reshape(TD.shape[0],1)
  TD_times_X = np.multiply(TD, X)
  
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
  
def update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat):
  Xw = np.dot(X, w)
  theta = update_theta(theta, alpha, gamma, TD_times_X, Xw, X_hat)
  w = update_w(w, beta, TD_times_X, X, Xw)
  return theta, w

def GGQ_step(theta, w, alpha, beta, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, X, intercept):
  TD_times_X, X_hat = GGQ_pieces(theta, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, X, intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat)
  return theta, w

def GGQ(rollout_feature_list, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept=True):
  N_IT = 20
  NU = 1/20
  
  nFeature = len(rollout_Q_function_list) + intercept
  theta, w = np.zeros(nFeature), np.zeros(nFeature)
  X = np.vstack([rollout_Q_features(data_block, rollout_Q_function_list, intercept) for data_block in env.X[:-1]])

  for it in range(N_IT):
    alpha = NU / ((it + 1) * np.log(it + 2))
    beta  = NU / (it + 1)
    theta, w = GGQ_step(theta, w, alpha, beta, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, X, intercept=intercept)
    #print('theta: {}'.format(theta))
  return theta
  
  
  
  