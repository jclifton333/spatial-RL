# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:53:17 2018

@author: Jesse

"""

import numpy as np
from lookahead import Q_max_all_states
from scipy.optimize import minimize
import pdb

def Q_from_rollout_features(data_block, theta, rollout_feature_list, rollout_Q_function_list):
  rollout_Q_features = np.array([q(data_block) for q in rollout_Q_function_list]).T
  rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return np.dot(rollout_Q_features, theta)
     
def QL_objective(theta, rollout_feature_list, rollout_Q_function_list, gamma, 
                 env, evaluation_budget, treatment_budget):   
  
  Q_fn = lambda data_block: Q_from_rollout_features(data_block, theta, rollout_feature_list, rollout_Q_function_list)
  Q = np.array([np.sum(Q_fn(data_block)) for data_block in env.X])
  Qmax, _, _ = Q_max_all_states(env, evaluation_budget, treatment_budget, Q_fn)
  Qmax = np.sum(Qmax, axis=0)
  TD = env.R[:-1] + gamma*Qmax - Q
  return np.dot(TD, TD)

def Qopt(rollout_feature_list, rollout_Q_function_list, gamma, 
         env, evaluation_budget, treatment_budget):
  objective = lambda theta: QL_objective(theta, rollout_feature_list, rollout_Q_function_list, 
                            gamma, env, evaluation_budget, treatment_budget)
  soln = minimize(objective, x0=np.zeros(len(rollout_Q_function_list) + 1), method='L-BFGS-B')
  return soln.x

'''
GGQ implementation
See: http://old.sztaki.hu/~szcsaba/papers/ICML10_controlGQ.pdf

'''

def theta_update(theta, alpha, delta, phi, gamma, phi_hat, w_dot_phi):
  theta += alpha * (delta * phi - gamma * (w_dot_phi * phi_hat))
  return theta

def w_update(beta, delta, phi, w, w_dot_phi):
  w += beta * (delta - w_dot_phi) * phi

def GGQ_update(theta, w, alpha, delta, phi, gamma, phi_hat):
  w_dot_phi = np.dot(w, phi)
  theta = theta_update(theta, alpha, delta, phi, gamma, phi_hat, w_dot_phi)
  w = w_update(beta, delta, phi, w, w_dot_phi)
  
  
  
  