# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:01 2018

@author: Jesse

Various auto-regressive classifiers for disease spread.
"""

import numpy as np
import pdb


def data_block_at_action(model, t, action, feature_function, predicted_probs):
  data_block = np.column_stack((model.S[t,:], action, model.Y[t,:]))
  if predicted_probs is not None:
    data_block = np.column_stack((predicted_probs, data_block))
  data_block = feature_function(data_block)
  return data_block
    
class AutoRegressor(object):
  '''
  Predict 1-step infection probabilities or k-step Q-values using neighbors' 1-step infection probabilities as features
  (thus 'auto').
  '''
  
  def __init__(self, ar_classifier, uc_classifier, regressor, featureFunction):
    '''
    :param ar_classifier: Model family to be used for autoregressive 1-step infected/not infected classification (e.g. RandomForestClassifier).
    :param uc_classifier: Model family for unconditional 1-step infected/notinfected classification.
    :param regressor:  Model family to be used for Q-fn regressions (e.g. RandomForestRegressor).
    :param featureFunction: Function that accepts raw [S, A, Y] and returns features.  
    '''
    
    self.uc_classifier = uc_classifier
    self.ar_classifier = ar_classifier
    self.regressor  = regressor
    self.featureFunction = featureFunction
    self.autoRegressionReady = False
    
  def unconditionalLogit(self, env):
    '''
    Fit unconditional one-step presence/absence logit on current data.
    '''
    self.uc_classifier.fit(np.vstack(env.X), np.hstack(env.y))
    self.predict_uc = lambda data_block: self.uc_classifier.predict_proba(data_block)[:,-1]
    self.pHat_uc = np.array([self.predict_uc(env.X[t]) for t in range(env.T)])

  def autoRegressionFeatures(self, env):
    '''
    Create autoregression dataset by appending sums of neighbor (unconditional) predicted probabilities 
    to unconditional dataset.
    '''    
    T = env.T
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
    self.X_ac = np.column_stack((pHat_sums, np.vstack(env.X)))
    self.X_ac = self.featureFunction(self.X_ac)   
    
  def createAutoregressionDataset(self, env):
    self.unconditionalLogit(env)
    self.autoRegressionFeatures(env)
    self.autoRegressionReady = True
    
  def createAutologitPredictor(self, binary):
    '''
    Sets function that returns predictions from fitted autologit model for a given data block.
    '''
    def autologitPredictor(dataBlock):
      uc_predictions = self.uc_classifier.predict_proba(dataBlock)[:,1]
      autologitDataBlock = np.column_stack((uc_predictions, dataBlock))
      if binary:
        predictions = self.ar_classifier.predict_proba(autologitDataBlock)[:,-1]
      else:
        predictions = self.regressor.predict(autologitDataBlock)
      return predictions
    self.autologitPredictor = autologitPredictor
    
  def fitClassifier(self, target):
    assert self.autoRegressionReady
    self.ar_classifier.fit(self.X_ac, target)
    self.createAutologitPredictor(binary=True)
    
  def fitRegressor(self, target):
    assert self.autoRegressionReady
    self.regressor.fit(self.X_ac, target)
    self.createAutologitPredictor(binary=False)
    
  


    
    
    
    
    


  
  
