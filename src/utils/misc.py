import numpy as np
from sklearn.linear_model import Ridge
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.regularizers import L1L2


def onehot(length, ix):
  arr = np.zeros(length)
  arr[ix] = 1
  return arr


def random_argsort(arr, num_to_take):
  """
  Ad-hoc way of getting randomized argsort.
  """
  top_entries = np.sort(-arr)[:(num_to_take*2)]
  b = np.random.random(top_entries.size)
  return np.argsort(np.lexsort((b, top_entries)))[:num_to_take]


class RidgeProb(object):
  def __init__(self):
    self.reg = Ridge()

  def fit(self, X, y):
    self.reg.fit(X, y)

  def predict_proba(self, X):
    phat = self.reg.predict(X)
    return np.column_stack((1-phat, phat))


class KerasLogit(object):
  def __init__(self):
    self.reg = Sequential()

  def fit(self, X, y):
    input_shape = X.shape[1]
    self.reg.add(Dense(1,
                       activation='softmax',
                       kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                       input_dim=input_shape))
    self.reg.compile(optimizer='sgd',
                     loss='binary_crossentropy')
    self.reg.fit(X, y)

  def predict_proba(self, X):
    phat = self.reg.predict(X)
    return np.column_stack((1-phat, phat))


