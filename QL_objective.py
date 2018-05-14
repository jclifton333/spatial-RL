# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse

"""

import numpy as np
from lookahead import Q_max_all_states
from scipy.optimize import minimize
import pdb

def rollout_Q_features(data_block, rollout_Q_function_list, intercept):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features

'''
GGQ implementation
See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf
'''

def GGQ_pieces(theta, rollout_Q_function_list, gamma, 
                 env, evaluation_budget, treatment_budget, intercept=True):   
  
  rollout_Q_features_at_block = lambda data_block: rollout_Q_features(data_block, rollout_Q_function_list, intercept)
  
  #Evaluate Q 
  X = np.array([rollout_Q_features_at_block(data_block) for data_block in env.X])
  Q = np.dot(X, theta)
  
  #Get Qmax  
  Q_fn = lambda data_block: np.dot(rollout_Q_features_at_block(data_block), theta)
  Qmax, Qargmax, _ = Q_max_all_states(env, evaluation_budget, treatment_budget, Q_fn)
  Qmax = np.sum(Qmax, axis=0)
  X_hat = np.array([rollout_Q_features_at_block(x) for x in Qargmax])
  
  #Compute TD * semi-gradient
  TD = env.R[:-1] + gamma*Qmax - Q
  TD = TD.reshape(len(TD),1)
  TD_times_X = np.multiply(TD, X)
  
  return TD_times_X, X, X_hat

def update_theta(theta, alpha, gamma, N, TD_times_X, Xw, X_hat):
  Xw_times_Xhat = np.multiply(Xw.reshape(len(Xw), 1), X_hat)
  gradient_block = TD_times_X - gamma*Xw_times_Xhat
  theta += alpha * (1 / N) * np.sum(gradient_block, axis=0)
  return theta

def update_w(w, beta, N, TD_times_X, X, Xw):
  Xw_times_X = np.multiply(Xw.reshape(len(Xw), 1), X)
  gradient_block = TD_times_X - Xw_times_X
  w += beta * (1 / N) * np.sum(gradient_block, axis=0)
  return w
  
def update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat):
  N = TD_times_X.shape[0]
  Xw = np.dot(X, w)
  theta = update_theta(theta, alpha, gamma, N, TD_times_X, Xw, X_hat)
  w = update_w(w, beta, N, TD_times_X, X, Xw)
  return theta, w

def GGQ_step(theta, w, alpha, beta, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept):
  TD_times_X, X, X_hat = GGQ_pieces(theta, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept)
  theta, w = update_theta_and_w(theta, w, alpha, beta, gamma, TD_times_X, X, X_hat)
  return theta, w

def GGQ(rollout_feature_list, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept=True):
  N_IT = 100
  
  nFeature = len(rollout_Q_function_list) + intercept
  theta, w = np.zeros(nFeature), np.zeros(nFeature)
  for it in range(N_IT):
    alpha = (it + 1)**(-1)
    beta  = (it + 1)**(-2/3)
    theta, w = GGQ_step(theta, w, alpha, beta, rollout_Q_function_list, gamma, env, evaluation_budget, intercept=intercept)
  return theta, w
  
  
  
  