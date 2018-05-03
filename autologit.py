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


def data_block_at_action(model, t, action, feature_function, predicted_probs):
  data_block = np.column_stack((model.S[t,:], action, model.Y[t,:]))
  if predicted_probs is not None:
    data_block = np.column_stack((predicted_probs, data_block))
  data_block = feature_function(data_block)
  return data_block
  
def create_CRF_dataset(model):
  #Create edge list
  edges = np.array([[i,j] for i in range(model.nS) for j in range(model.nS) if model.adjacency_matrix[i,j] == 1])
  
  #Create Xs and ys
  X = [(np.column_stack((model.A[t,:], model.Y[t,:])), edges) for t in range(model.T - 1)]
  y = [model.Y[t,:].astype(int) for t in range(1, model.T)]
  
  return X, y

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
  :return predicted_probs_list: length-T list whose t-th entry is corresponding vector of predicted_prob_sums
  '''
  T = model.A.shape[0]
  predicted_prob_sums = np.array([])
  predicted_probs_list = []
  for t in range(T):
    unconditional_predicted_probs_t = unconditional_predicted_probs[t,:]
    predicted_prob_sums_t = np.array([])
    for l in range(model.nS):
      neighbor_predicted_probs_l = unconditional_predicted_probs_t[model.adjacency_list[l]]
      predicted_prob_sums_t = np.append(predicted_prob_sums_t, np.sum(neighbor_predicted_probs_l))
    predicted_probs_list.append(predicted_prob_sums_t)
    predicted_prob_sums = np.append(predicted_prob_sums, predicted_prob_sums_t)
  data = np.column_stack((predicted_prob_sums, unconditional_dataset))
  return data, predicted_probs_list

def unconditional_logit(model, classifier, data, target, binary):
  #Logistic regression
  logit = classifier(oob_score=True, n_estimators=30)
  logit.fit(data, target)
  if binary:
    phat = logit.predict_proba(data)[:,-1]
    predict = lambda data_block: logit.predict_proba(data_block)[:,-1]
  else:
    phat = logit.predict(data)
    predict = lambda data_block: logit.predict(data_block)
    print('Score: {}'.format(logit.oob_score_))
  return predict, phat.reshape((model.A.shape[0], model.nS))
  
def autologit(model, classifier, unconditional_classifier, unconditional_data, target, binary=True, predicted_probs=None, uc_logit=None):
  if predicted_probs is None:
    uc_logit, predicted_probs = unconditional_logit(model, unconditional_classifier, unconditional_data, target, binary = binary)
  autologit_data, predicted_probs_list = create_autologit_dataset(model, unconditional_data, predicted_probs)

  logit = classifier(oob_score=True, n_estimators=30)
  logit.fit(autologit_data, target)
  print(logit.oob_score_)

  if binary:
    predictions = logit.predict_proba(autologit_data)[:,-1]
    predict = lambda data_block: logit.predict_proba(data_block)[:,-1]
  else:
    predictions = logit.predict(autologit_data)
    predict = lambda data_block: logit.predict(data_block)
    
  def autologit_predictor(data_block):
    uc_predictions = uc_logit(data_block)
    autologit_data_block = np.column_stack((uc_predictions, data_block))
    predictions = predict(autologit_data_block)
    return predictions    
    
  return autologit_predictor, uc_logit, predictions.reshape((model.A.shape[0], model.nS)), predicted_probs_list, predicted_probs, autologit_data
  
class AutoRegressor(object):
  '''
  Predict 1-step infection probabilities or k-step Q-values using neighbors' 1-step infection probabilities as features
  (thus 'auto').
  '''
  
  def __init__(self, classifier, regressor, featureFunction):
    '''
    :param classifier: Model family to be used for 1-step infected/not infected classification (e.g. RandomForestClassifier).
    :param regressor:  Model family to be used for Q-fn regressions (e.g. RandomForestRegressor).
    :param featureFunction: Function that accepts raw [S, A, Y] and returns features.  
    '''
    
    self.uc_classifier = classifier
    self.ar_classifier = classifier
    self.regressor  = regressor
    self.featureFunction = featureFunction
    
  def unconditionalData(self, env):
    '''
    Create targets and unconditional features.      
    :param env: Disease model environment with S, A, and Y array attributes. 
    '''
    T = env.A.shape[0]
    X = np.column_stack(env.S[0,:], env.A[0,:], env.Y[0,:])
    y = env.Y[1,:]
    for t in range(1, T):
      X_block = np.column_stack(env.S[t,:], env.A[t,:], env.Y[t,:])
      y_block = env.Y[t+1,:]
      X = np.vstack((X, X_block))
      y = np.append(y, y_block)
    X = self.featureFunction(X)
    self.X_uc = X
    self.y = y.astype(float)
        
  def unconditionalLogit(self):
    '''
    Fit unconditional one-step presence/absence logit on current data.
    '''
    self.uc_classifier.fit(self.X_uc, self.y)
    self.pHat_uc = np.array([self.uc_classifier.predict_proba(self.X_uc_blocks[t,:]) for t in range(self.X_uc_blocks.shape[0])][:,-1])
    self.predict_uc = lambda data_block: self.uc_classifier.predict(data_block)[:,-1]

  def autoRegressionFeatures(self, env):
    '''
    Create autoregression dataset by appending sums of neighbor (unconditional) predicted probabilities 
    to unconditional dataset.
    '''    
    T = env.A.shape[0]
    pHat_sums = np.array([])
    pHat_list = []
    for t in range(T):
      pHat_uc_t = self.pHat_uc[t,:]
      pHat_sums_t = np.array([])
      for l in range(env.nS):
        neighbor_predicted_probs_l = pHat_uc_t[env.adjacency_list[l]]
        pHat_sums_t = np.append(pHat_sums_t, np.sum(neighbor_predicted_probs_l))
      pHat_list.append(pHat_sums_t)
      pHat_sums = np.append(pHat_sums, pHat_sums_t)
    self.X_ac = np.column_stack((pHat_sums, self.X_uc))
    self.X_ac = self.featureFunction(self.X_ac)   
    
  def createAutoregressionDataset(self, env):
    self.unconditionalData(env)
    self.unconditionalLogit()
    self.autoRegressionFeatures(env)
    
  def fit(self, env, target):
    self.createAutoregressionDataset(env)
    self.regressor.fit(self.X_ac, target)
  
    
    
    
    
    
    


  
  
