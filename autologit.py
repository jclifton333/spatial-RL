# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:01 2018

@author: Jesse

Various auto-regressive classifiers for disease spread.
"""

import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
#import pdb

def create_unconditional_dataset(model):
  '''
  Create dataset to model unconditional probability of disease.  
  :param model: disease model object
  :return data: 2d array with columns [state, action, infections, next-infections]
  '''
  T = model.A.shape[0]
  for t in range(T):
    data_block = np.column_stack((model.S[t,:], model.A[t,:], model.Y[t,:], model.Y[t+1,:]))
    if t == 0:
      data = data_block
    else:
      data = np.vstack((data, data_block))
  return data

def create_autologit_dataset(model, unconditional_dataset, unconditional_predicted_probs):
  '''
  Create dataset appropriate for autologistic model.
  :param model: disease model object
  :param unconditional_dataset: 2d-array used to estimate unconditional probabilities of infection
  :param unconditional_predicted_probs: unconditional predicted probabilities for each location at each timestep
  :return data: 2d array, autologistic sums appended to unconditional dataset
  '''
  T = model.A.shape[0]
  predicted_prob_sums = np.array([])
  for t in range(T):
    unconditional_predicted_probs_t = unconditional_predicted_probs[t,:]
    for l in range(model.nS):
      neighbor_predicted_probs_l = unconditional_predicted_probs_t[model.adjacency_list[l]]
      predicted_prob_sums = np.append(predicted_prob_sums, np.sum(neighbor_predicted_probs_l))
  data = np.column_stack((predicted_prob_sums, unconditional_dataset))
  return data

def unconditional_logit(model, classifier):
  data = create_unconditional_dataset(model)
  
  #Create interactions 
  state_action_ixn = np.multiply(data[:,0], data[:,1])
  state_infect_ixn = np.multiply(data[:,0], data[:,2])
  action_infect_ixn = np.multiply(data[:,1], data[:, 2])
  state_action_infect_ixn = np.multiply(data[:,0], action_infect_ixn)
  data = np.column_stack((state_action_ixn, state_infect_ixn, action_infect_ixn, state_action_infect_ixn, data))
  
  #Logistic regression
  logit = classifier()
  logit.fit(data[:,:-1], data[:,-1])
  phat = logit.predict_proba(data[:,:-1])
  return(data, phat[:,-1].reshape((model.A.shape[0], model.nS)))
  
def autologit(model, classifier, unconditional_classifier):
  unconditional_data, predicted_probs = unconditional_logit(model, unconditional_classifier)
  data = create_autologit_dataset(model, unconditional_data, predicted_probs)
  logit = classifier()
  logit.fit(data[:,:-1], data[:,-1])
  predictions = logit.predict_proba(data[:,:-1])
  return logit, predictions[:,-1].reshape((model.A.shape[0], model.nS))
  


  
  
