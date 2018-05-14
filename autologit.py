# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:51:01 2018

@author: Jesse

Various auto-regressive classifiers for disease spread.
"""

import numpy as np
import pdb
    
class AutoRegressor(object):
  '''
  Predict 1-step infection probabilities or k-step Q-values using neighbors' 1-step infection probabilities as features
  (thus 'auto').
  '''
  
  def __init__(self, ar_classifier, regressor):
    '''
    :param ar_classifier: Model family to be used for autoregressive 1-step infected/not infected classification (e.g. RandomForestClassifier).
    :param regressor:  Model family to be used for Q-fn regressions (e.g. RandomForestRegressor).
    '''
    
    self.ar_classifier = ar_classifier
    self.regressor  = regressor
        
  def createAutologitPredictor(self, binary, env):
    '''
    Sets function that returns predictions from fitted autologit model for a given data block.
    '''
    def autologitPredictor(dataBlock):
      #Fit UC predictions if not already provided
      if binary:
        predictions = self.ar_classifier.predict_proba(dataBlock)[:,-1]
      else:
        predictions = self.regressor.predict(dataBlock)        
      return predictions    
    self.autologitPredictor = autologitPredictor
    
  def fitClassifier(self, env, target):
    self.ar_classifier.fit(np.vstack(env.X), target)
    self.createAutologitPredictor(True, env)
    
  def fitRegressor(self, env, target):
    self.regressor.fit(np.vstack(env.X), target)
    self.createAutologitPredictor(False, env)
    
  


    
    
    
    
    


  
  
