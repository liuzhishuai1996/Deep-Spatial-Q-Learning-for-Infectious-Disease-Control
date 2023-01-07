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
    :param ar_classifier: Model family to be used for autoregressive 1-step infected/not infected classification
                         (e.g. RandomForestClassifier).
    :param regressor:  Model family to be used for Q-fn regressions (e.g. RandomForestRegressor).
    '''
    
    self.ar_classifier = ar_classifier()
    self.regressor = regressor()
    self.autologitPredictor = None
    self.predictors = []  # For storing sequence of fitted-Q functions (for use as features in QL); not currently used
        
  def resetPredictors(self, bootstrap):
    self.predictors = []
    
  def createAutologitPredictor(self, predictor, addToList, binary):
    '''
    Sets function that returns predictions from fitted autologit model for a given data block.
    '''
    def autologitPredictor(dataBlock):
      # Fit UC predictions if not already provided
      if binary:
        predictions = predictor.predict_proba(dataBlock)[:, -1]
      else:
        predictions = predictor.predict(dataBlock)        
      return predictions    
    if addToList: self.predictors.append(autologitPredictor)
    self.autologitPredictor = autologitPredictor
    
  def fitClassifier(self, features, target, weights, addToList, exclude_neighbor_sums):
    self.ar_classifier.fit(features, target, weights, exclude_neighbor_sums)
    self.createAutologitPredictor(self.ar_classifier, addToList, binary=True)
    
  def fitRegressor(self, features, target, weights, addToList):
    self.regressor.fit(features, target, weights)
    self.createAutologitPredictor(self.regressor, addToList, binary=False)
    
  


    
    
    
    
    


  
  
