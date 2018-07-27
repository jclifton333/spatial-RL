import pdb
from functools import partial
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2
import keras.backend as K


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


def bootstrap_sq_error_loss(yTrue, yPred, weights):
  return K.mean( weights * K.square(yTrue - yPred))


class KerasRegressor(object):
  def __init__(self):
    self.reg = Sequential()

  def fit(self, X, y, weights, grid_search=True):
    input_shape = X.shape[1]
    self.reg.add(Dense(int(np.floor(input_shape/2)), input_dim=input_shape,
                 activation='relu', kernel_regularizer=L1L2(l1=0.0, l2=100)))
    self.reg.add(Dense(1, kernel_regularizer=L1L2(l1=0.0, l2=100)))
    if weights is not None:
      loss = partial(bootstrap_sq_error_loss, weights=weights)
    else:
      loss = 'mean_squared_error'
    self.reg.compile(optimizer='adam', loss=loss)
    self.reg.fit(X, y, verbose=0)

  def predict(self, X):
    return self.reg.predict(X).reshape(-1)


# ToDo: SKLogit and KerasLogit inherit from Logit superclass
class KerasLogit(object):
  def __init__(self):
    self.reg = Sequential()
    self.intercept_ = None
    self.coef_ = None
    self.fitted_model = False
    self.exclude_neighbor_features = False  # This should be set to true if env.add_neighbor_features=True
    self.input_shape = None

  def fit_keras(self, X, y, weights):
    self.reg.add(Dense(1,
                       activation='sigmoid',
                       kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                       input_dim=self.input_shape))
    self.reg.compile(optimizer='rmsprop', loss='binary_crossentropy')

    self.reg.fit(X, y, sample_weight=weights, epochs=5)
    self.get_coef()

  def fit(self, X, y, weights, exclude_neighbor_features=False):
    self.input_shape = X.shape[1]
    # Cut X in half if exclude neighbor features
    if exclude_neighbor_features:
      self.exclude_neighbor_features = True
      self.input_shape = int(self.input_shape / 2)  # adding neighbor features doubles number of features
      X = X[:, self.input_shape]
    y0 = y[0]
    for element in y:
      if element == 1 - y0:
        self.fit_keras(X, y, weights)
        self.fitted_model = True
        return
    # Hacky way of dealing with all-0 or all-1 targets
    self.intercept_ = -0.001 + y0
    self.coef_ = -0.001 + np.zeros(self.input_shape + 1)

  def get_coef(self):
    """
    Keras stores info for each layer in list reg.layers, and each layer object has method get_weights(), which returns
    list [hidden_layer_coefficient_array, hidden_layer_bias_array].
    :return:
    """
    coef_list = self.reg.layers[0].get_weights()
    self.intercept_ = coef_list[1]
    self.coef_ = coef_list[0]

  def predict_proba(self, X):
    if self.fitted_model:
      if self.exclude_neighbor_features:
        X = X[:, :self.input_shape]
      phat = self.reg.predict_proba(X)
      return np.column_stack((1-phat, phat))
    else:
      return np.column_stack((np.ones(X.shape[0]), np.zeros(X.shape[0])))


class SKLogit(object):
  def __init__(self):
    self.reg = LogisticRegression(C=1.0 / 0.1)
    self.intercept_ = None
    self.coef_ = None

  def fit(self, X, y, weights):
    y0 = y[0]
    for element in y:
      if element == 1 - y0:
        self.reg.fit(X, y, sample_weight=weights)
        self.fitted_model = True
        self.get_coef()
        return
    # Hacky way of dealing with all-0 or all-1 targets
    self.intercept_ = -0.001 + y0
    self.coef_ = -0.001 + np.zeros(X.shape[1] + 1)

  def get_coef(self):
    self.intercept_ = self.reg.intercept_
    self.coef_ = self.reg.coef_

  def predict_proba(self, X):
    if self.fitted_model:
      phat = self.reg.predict_proba(X)
      return phat
    else:
      return np.column_stack((np.ones(X.shape[0]), np.zeros(X.shape[0])))


