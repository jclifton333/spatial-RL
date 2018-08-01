import pdb
import numpy as np
from scipy.special import expit


class SIS_Contaminator(object):
  def __init__(self):
    self.weights = None

  def set_weights(self, new_weights, n_feature):
    self.weights = new_weights

  def predict_proba(self, X):
    if self.weights is None:
      self.weights = np.random.normal(size=(X.shape[1] + 1))
    logit_p = np.dot(np.column_stack((X, np.ones(X.shape[0]))), self.weights)
    p = expit(logit_p)
    return np.column_stack((1-p, p))
