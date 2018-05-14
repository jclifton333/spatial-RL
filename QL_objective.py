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
  rollout_Q_features = np.array([np.sum(q(data_block)) for q in rollout_Q_function_list]).T
  if intercept:
    rollout_Q_features = np.column_stack((np.ones(rollout_Q_features.shape[0]), rollout_Q_features))
  return rollout_Q_features

#def Qopt(rollout_feature_list, rollout_Q_function_list, gamma, 
#         env, evaluation_budget, treatment_budget):
#  objective = lambda theta: QL_objective(theta, rollout_feature_list, rollout_Q_function_list, 
#                            gamma, env, evaluation_budget, treatment_budget)
#  soln = minimize(objective, x0=np.zeros(len(rollout_Q_function_list) + 1), method='L-BFGS-B')
#  return soln.x

'''
Semi-gradient QL opt implementation
'''
def compute_QL_semi_gradient(theta, X, rollout_Q_function_list, gamma, 
                 env, evaluation_budget, treatment_budget, intercept=True):   
  
  #Evaluate Q 
  X = np.array(rollout_feature_list)
  Q = np.dot(X, theta)
  
  #Get Qmax  
  rollout_Q_features_at_block = lambda data_block: rollout_Q_features(data_block, rollout_Q_function_list, intercept)
  Q_fn = lambda data_block: np.dot(rollout_Q_features_at_block(data_block), theta)
  Qmax, _, _ = Q_max_all_states(env, evaluation_budget, treatment_budget, Q_fn)
  Qmax = np.sum(Qmax, axis=0)
  
  #Compute TD * semi-gradient
  TD = env.R[:-1] + gamma*Qmax - Q
  TD_times_grad = np.multiply(TD, X)
  return np.sum(TD_times_grad, axis=0)

def QL_opt_semi_gradient(rollout_features_list, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept):
  NUM_IT = 100
  
  X = np.array(rollout_features_list)
  if intercept:
    pdb.set_trace()
    X = np.column_stack((np.ones(X.shape[0], X)))
  theta = np.zeros(X.shape[1])
  for it in range(NUM_IT):
    gradTheta = compute_QL_semi_gradient(theta, X, rollout_Q_function_list, gamma, env, evaluation_budget, treatment_budget, intercept=intercept)
    theta -= (it+1)**(0.75) * gradTheta
  return theta  

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
  
  
  
  