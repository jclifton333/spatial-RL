import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import L1L2


class RidgeProb(object):
  def __init__(self):
    self.reg = Ridge()

  def fit(self, X, y):
    self.reg.fit(X, y)

  def predict_proba(self, X):
    phat = self.reg.predict(X)
    return np.column_stack((1-phat, phat))


class KerasRegressor(object):
  def __init__(self):
    self.reg = Sequential()

  def fit(self, X, y, weights):
    input_shape = X.shape[1]
    self.reg.add(Dense(int(np.floor(input_shape/2)), input_dim=input_shape,
                 activation='relu', kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    self.reg.add(Dense(1, kernel_regularizer=L1L2(l1=0.0, l2=0.1)))
    self.reg.compile(optimizer='adam', loss='mean_squared_error')
    self.reg.fit(X, y, sample_weight=weights, verbose=True)

  def predict(self, X):
    return self.reg.predict(X).reshape(-1)


def is_y_all_1_or_0(y):
    y0 = y[0]
    for element in y:
      if element == 1 - y0:
        return False
    return True


class SKLogit(object):
  def __init__(self):
    self.reg = LogisticRegression()
    self.condition_on_infection = False
    self.intercept_ = None
    self.coef_ = None

  def fit(self, X, y, weights):
    if is_y_all_1_or_0(y):
      y0 = y[0]
      n = len(y)
      expit_intercept_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
      self.intercept_ = logit(expit_intercept_)
      self.coef_ = np.zeros(X.shape[1])
    else:
      self.reg.fit(X, y, sample_weight=weights)

  def get_coef(self):
    self.intercept_ = self.reg.intercept_
    self.coef_ = self.reg.coef_

  def predict_proba(self, X):
    if self.fitted_model:
      phat = self.reg.predict_proba(X)
      return phat
    else:
      return np.column_stack((np.ones(X.shape[0]), np.zeros(X.shape[0])))


class SKLogit2(object):
  def __init__(self):
    self.reg_inf = LogisticRegression()
    self.reg_not_inf = LogisticRegression()
    self.condition_on_infection = True
    self.fitted_model = False

  def fit(self, X, y, weights, infected_locations, not_infected_locations):
    if is_y_all_1_or_0(y):
      pass
    else:
      if weights is not None:
        inf_weights = weights[infected_locations]
        not_inf_weights = weights[not_infected_locations]
      else:
        inf_weights = not_inf_weights = None
      self.reg_inf.fit(X[infected_locations], y[infected_locations], sample_weight=inf_weights)
      self.reg_not_inf.fit(X[not_infected_locations], y[not_infected_locations],
                           sample_weight=not_inf_weights)
      self.fitted_model = True

  def predict_proba(self, X, infected_locations, not_infected_locations):
    if self.fitted_model:
      phat = np.zeros((X.shape[0], 2))
      phat[infected_locations] = self.reg_inf.predict_proba(X[infected_locations])
      phat[not_infected_locations] = self.reg_not_inf.predict_proba(X[not_infected_locations])
      return phat
    else:
      return np.column_stack((np.ones(X.shape[0]), np.zeros(X.shape[0])))


# #
# class KerasLogit(object):
#   def __init__(self):
#     self.reg = Sequential()
#     self.intercept_ = None
#     self.coef_ = None
#     self.fitted_model = False
#     self.exclude_neighbor_features = False  # This should be set to true if env.add_neighbor_features=True
#     self.input_shape = None
#     self.layer_added = False
#
#   def set_weights(self, new_weights, n_feature):
#     """
#
#     :param new_weights: [coef array, bias array]
#     :return:
#     """
#     self.fitted_model = True
#     if self.reg.layers:
#       self.reg.layers[0].set_weights(new_weights)
#     else:
#       self.reg.add(Dense(1,
#                          activation='sigmoid',
#                          kernel_regularizer=L1L2(l1=0.0, l2=0.1),
#                          input_dim=n_feature,
#                          weights=new_weights))
#       self.layer_added = True
#
#   def fit_keras(self, X, y, weights):
#     if not self.layer_added:
#       self.reg.add(Dense(1,
#                          activation='sigmoid',
#                          kernel_regularizer=L1L2(l1=0.0, l2=0.1),
#                          input_dim=self.input_shape))
#       self.layer_added = True
#     self.reg.compile(optimizer='rmsprop', loss='binary_crossentropy')
#     self.reg.fit(X, y, sample_weight=weights, epochs=5, verbose=False)
#     self.get_coef()
#
#   def fit(self, X, y, weights, exclude_neighbor_features=False):
#     self.input_shape = X.shape[1]
#     # Cut X in half if exclude neighbor features
#     if exclude_neighbor_features:
#       self.exclude_neighbor_features = True
#       self.input_shape = int(self.input_shape / 2)  # adding neighbor features doubles number of features
#       X = X[:, :self.input_shape]
#     y0 = y[0]
#     for element in y:
#       if element == 1 - y0:
#         self.fit_keras(X, y, weights)
#         self.fitted_model = True
#         return
#     # Hacky way of dealing with all-0 or all-1 targets
#     self.intercept_ = -0.001 + y0
#     self.coef_ = -0.001 + np.zeros(self.input_shape)
#
#   def get_coef(self):
#     """
#     Keras stores info for each layer in list reg.layers, and each layer object has method get_weights(), which returns
#     list [hidden_layer_coefficient_array, hidden_layer_bias_array].
#     :return:
#     """
#     coef_list = self.reg.layers[0].get_weights()
#     self.intercept_ = coef_list[1]
#     self.coef_ = coef_list[0]
#
#   def predict_proba(self, X):
#     if self.fitted_model:
#       if self.exclude_neighbor_features:
#         X = X[:, :self.input_shape]
#       phat = self.reg.predict_proba(X)
#       return np.column_stack((1-phat, phat))
#     else:
#       return np.column_stack((np.ones(X.shape[0]), np.zeros(X.shape[0])))
