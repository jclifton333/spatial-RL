# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:01 2018

@author: Jesse

Various auto-regressive classifiers for disease spread.
"""

import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
import pdb


def data_block_at_action(model, t, action, feature_function):
  data_block = np.column_stack((model.S[t,:], action, model.Y[t,:]))
  data_block = feature_function(data_block)
  return data_block

def create_unconditional_dataset(model, feature_function):
  '''
  Create dataset to model unconditional probability of disease.  
  :param model: disease model object
  :param feature_function: function that returns addition features of [state, action, infections]
  :return data: 2d array with columns [state, action, infections, next-infections]
  '''
  T = model.A.shape[0]
  for t in range(T):
    data_block = np.column_stack((model.S[t,:], model.A[t,:], model.Y[t,:]))
    target_block = model.Y[t+1,:]
    if t == 0:
      data = data_block
      target = target_block
    else:
      data = np.vstack((data, data_block))
      target = np.append(target, target_block)
  data = feature_function(data)
  return data, target.astype(float)
 
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

def unconditional_logit(model, classifier, data, target, binary):
  #Logistic regression
  logit = classifier()
  logit.fit(data[:,:-1], target)
  if binary:
    phat = logit.predict_proba(data[:,:-1])[:,-1]
  else:
    phat = logit.predict(data[:,:-1])
  return phat.reshape((model.A.shape[0], model.nS))
  
def autologit(model, classifier, unconditional_classifier, unconditional_data, target, binary=True):
  predicted_probs = unconditional_logit(model, unconditional_classifier, unconditional_data, target, binary = binary)
  autologit_data = create_autologit_dataset(model, unconditional_data, predicted_probs)
  logit = classifier()
  logit.fit(autologit_data[:,:-1], target)
  if binary:
    predictions = logit.predict_proba(autologit_data[:,:-1])[:,-1]
    predict = lambda data_block: logit.predict_proba(data_block)[:,-1]
  else:
    predictions = logit.predict(autologit_data[:,:-1])
    predict = lambda data_block: logit.predict(data_block)
  return predict, predictions.reshape((model.A.shape[0], model.nS)) 
  


  
  
