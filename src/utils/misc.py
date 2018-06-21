import numpy as np
from sklearn.linear_model import Ridge


def onehot(length, ix):
  arr = np.zeros(length)
  arr[ix] = 1
  return arr


class RidgeProb(object):
  def __init__(self):
    self.reg = Ridge()

  def fit(self, X, y):
    self.reg.fit(X, y)

  def predict_proba(self, X):
    phat = self.reg.predict(X)
    return np.column_stack((1-phat, phat))