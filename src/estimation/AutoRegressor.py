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
    self.predictors = []
        
  def resetPredictors(self):
    self.predictors = []
    
  def createAutologitPredictor(self, predictor, addToList, binary):
    '''
    Sets function that returns predictions from fitted autologit model for a given data block.
    '''
    def autologitPredictor(dataBlock):
      #Fit UC predictions if not already provided
      if binary:
        predictions = predictor.predict_proba(dataBlock)[:,-1]
      else:
        predictions = predictor.predict(dataBlock)        
      return predictions    
    if addToList: self.predictors.append(autologitPredictor)
    self.autologitPredictor = autologitPredictor
    
  def fitClassifier(self, env, target, addToList):
    classifier = self.ar_classifier()
    classifier.fit(np.vstack(env.X), target)
    self.createAutologitPredictor(classifier, addToList, binary=True)
    
  def fitRegressor(self, env, target, addToList):
    regressor = self.regressor()
    regressor.fit(np.vstack(env.X), target)
    self.createAutologitPredictor(regressor, addToList, binary=False)
    
  


    
    
    
    
    


  
  
