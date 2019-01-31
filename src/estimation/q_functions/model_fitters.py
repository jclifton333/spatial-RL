import pdb
import numpy as np
import src.utils.gradient as gradient
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.linalg import block_diag
from scipy.special import expit, logit
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import L1L2
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import talos as ta


class RidgeProb(object):
  def __init__(self):
    self.reg = Ridge()

  def fit(self, X, y):
    self.reg.fit(X, y)

  def predict_proba(self, X):
    phat = self.reg.predict(X)
    return np.column_stack((1-phat, phat))


def fit_piecewsie_keras_classifier(X, y, infected_indices, not_infected_indices):
  """
  FIt separate kears infection probability models for infected and non-infected locations.
  :param X:
  :param y:
  :param infected_indices:
  :param not_infected_indices:
  :return:

  """
  input_shape = X.shape[1]

  # Fit infected model
  reg = Sequential()
  reg.add(Dense(units=50, input_dim=input_shape, activation='relu'))
  reg.add(Dense(units=50, activation='relu'))
  reg.add(Dense(1, activation='sigmoid'))
  reg.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  reg.fit(X[infected_indices], y[infected_indices], sample_weight=None, verbose=True, epochs=5)

  # Fit not-infected model
  reg_nof_inf = Sequential()
  reg_nof_inf.add(Dense(units=50, input_dim=input_shape, activation='relu'))
  reg_nof_inf.add(Dense(units=50, activation='relu'))
  reg_nof_inf.add(Dense(1, activation='sigmoid'))
  reg_nof_inf.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  reg_nof_inf.fit(X[not_infected_indices], y[not_infected_indices], sample_weight=None, verbose=True, epochs=5)

  def predict_proba_piecewise(X_, infected_indices_, not_infected_indices_):
    probs = np.zeros(X_.shape[0])
    probs[infected_indices_] = reg.predict(X_[infected_indices]).flatten()
    probs[not_infected_indices_] = reg_nof_inf.predict(X_[not_infected_indices]).flatten()
    return probs

  # return reg, graph
  return predict_proba_piecewise


def fit_piecewise_keras_regressor(X, y, infected_indices, not_infected_indices):
  """
  Fit separate regression models for infected and not-infected locations.

  :param X:
  :param y:
  :param infected_indices:
  :param not_infected_indices:
  :return:
  """
  params = {
      'dropout1': 0.17,
      'dropout2': 0.17,
      'epochs': 20,
      'units1': 20,
      'units2': 10,
      'lr': 1.4
  }
  input_shape = X.shape[1]

  # Infected locations
  reg = Sequential()
  reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
  reg.add(Dropout(params['dropout1']))
  reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
  reg.add(Dropout(params['dropout2']))
  reg.add(Dense(1))
  reg.compile(optimizer='adam', loss='mean_squared_error')
  reg.fit(X[infected_indices], y[infected_indices], verbose=True, epochs=params['epochs'])

  # Not infected locations
  reg_not_inf = Sequential()
  reg_not_inf.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
  reg_not_inf.add(Dropout(params['dropout1']))
  reg_not_inf.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
  reg_not_inf.add(Dropout(params['dropout2']))
  reg_not_inf.add(Dense(1))
  reg_not_inf.compile(optimizer='adam', loss='mean_squared_error')
  reg_not_inf.fit(X[not_infected_indices], y[not_infected_indices], verbose=True, epochs=params['epochs'])

  def predict_piecewise(X_, infected_indices_, not_infected_indices_):
    predictions = np.zeros(X_.shape[0])
    predictions[infected_indices_] = reg.predict(X_[infected_indices]).flatten()
    predictions[not_infected_indices_] = reg_nof_inf.predict(X_[not_infected_indices]).flatten()
    return probs

  return predict_piecewise


def fit_keras_classifier(X, y):
  input_shape = X.shape[1]

  # graph = tf.Graph()
  # with graph.as_default():
  #   session = tf.Session()
  #   init = tf.global_variables_initializer()
  #   session.run(init)
  #   with session.as_default():
  #     reg = Sequential()
  #     reg.add(Dense(units=50, input_dim=input_shape, activation='relu'))
  #     reg.add(Dense(units=50, activation='relu'))
  #     reg.add(Dense(1, activation='sigmoid'))
  #     reg.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  #     reg.fit(X, y, sample_weight=None, verbose=True, epochs=5)

  reg = Sequential()
  reg.add(Dense(units=50, input_dim=input_shape, activation='relu'))
  reg.add(Dense(units=50, activation='relu'))
  reg.add(Dense(1, activation='sigmoid'))
  reg.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
  reg.fit(X, y, sample_weight=None, verbose=True, epochs=5)
  # return reg, graph
  return reg, None 


def fit_keras_regressor(X, y):
  params = {
      'dropout1': 0.17,
      'dropout2': 0.17,
      'epochs': 20,
      'units1': 20,
      'units2': 10,
      'lr': 1.4
  }
  input_shape = X.shape[1]

  # graph = tf.Graph()
  # with graph.as_default():
  #   session = tf.Session()
  #   init = tf.global_variables_initializer()
  #   session.run(init)
  #   with session.as_default():
  #     reg = Sequential()
  #     reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
  #     reg.add(Dropout(params['dropout1']))
  #     reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
  #     reg.add(Dropout(params['dropout2']))
  #     reg.add(Dense(1))
  #     reg.compile(optimizer='adam', loss='mean_squared_error')
  #     reg.fit(X, y, verbose=True, epochs=params['epochs'])

  reg = Sequential()
  reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
  reg.add(Dropout(params['dropout1']))
  reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
  reg.add(Dropout(params['dropout2']))
  reg.add(Dense(1))
  reg.compile(optimizer='adam', loss='mean_squared_error')
  reg.fit(X, y, verbose=True, epochs=params['epochs'])
  # return reg, graph
  return reg, None


class KerasRegressor(object):
  def __init__(self):
    pass

  def fit(self, X, y, hyperparameter_search=False, weights=None):
    input_shape = X.shape[1]
    self.hyperparameter_search = hyperparameter_search

    if hyperparameter_search:
      # Following https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
      params = {
       'dropout1': (0, 0.5, 3),
       'dropout2': (0, 0.5, 3),
       'epochs': [1, 20, 50],
       'units1': [20, 50, 100],
       'units2': [10, 25, 50],
       'lr': (0.5, 5, 5)
      }

      # Define model as function of grid params
      def model(X_train, y_train, X_val, y_val, params):
        reg = Sequential()
        reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
        reg.add(Dropout(params['dropout1']))
        reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
        reg.add(Dropout(params['dropout2']))
        reg.add(Dense(1))
        reg.compile(optimizer='adam', loss='mean_squared_error')
        history = reg.fit(X_train, y_train, verbose=True, epochs=params['epochs'],
                          validation_data=[X_val, y_val])
        return history, reg

      # Search
      self.t = ta.Scan(x=X, y=y, model=model, grid_downsample=0.1, params=params)

      # Get predictor (model corresponding to best hyperparameters)
      self.predictor = ta.Predict(self.t)

    else:
      # Params from hyperparameter search
      params = {
        'dropout1': 0.17,
        'dropout2': 0.17,
        'epochs': 20,
        'units1': 20,
        'units2': 10,
        'lr': 1.4
      }
      reg = Sequential()
      reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
      reg.add(Dropout(params['dropout1']))
      reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
      reg.add(Dropout(params['dropout2']))
      reg.add(Dense(1))
      reg.compile(optimizer='adam', loss='mean_squared_error')
      history = reg.fit(X, y, verbose=True, epochs=params['epochs'])
      K.clear_session()
      self.predictor = reg

  def predict(self, X):
    if self.hyperparameter_search:
      return self.predictor.predict(X, metric='val_loss').reshape(-1)
    else:
      return self.predictor.predict(X).reshape(-1)


def is_y_all_1_or_0(y):
    y0 = y[0]
    for element in y:
      if element == 1 - y0:
        return False
    return True


def empirical_bayes(y):
  y0 = y[0]
  n = len(y)
  expit_mean_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
  mean_ = logit(expit_mean_)
  return mean_


def empirical_bayes_coef(y, p):
  """
  For when y is all 1 or 0 and you need a coefficient vector.
  :param y:
  :param p:
  :return:
  """
  intercept_ = empirical_bayes(y)
  coef_ = np.zeros(p)
  return intercept_, coef_


class SKLogit(object):
  def __init__(self):
    self.reg = LogisticRegression()
    self.condition_on_infection = False
    self.intercept_ = None
    self.coef_ = None

  def fit(self, X, y, weights):
    if is_y_all_1_or_0(y):
      self.intercept_, self.coef_ = empirical_bayes_coef(y, X.shape[1])
    else:
      self.reg.fit(X, y, sample_weight=weights)
      self.get_coef()

  def get_coef(self):
    self.intercept_ = self.reg.intercept_
    self.coef_ = self.reg.coef_

  def predict_proba(self, X):
    phat = self.reg.predict_proba(X)
    return phat[:, -1]


class SKLogit2(object):
  def __init__(self):
    # self.reg_inf = RandomForestClassifier(200)
    # self.reg_not_inf = RandomForestClassifier(200)
    self.reg_inf = LogisticRegression()
    self.reg_not_inf = LogisticRegression()
    self.condition_on_infection = True
    self.inf_model_fitted = False
    self.not_inf_model_fitted = False
    self.inf_params = None
    self.not_inf_params = None
    self.inf_eb_prob = None
    self.not_inf_eb_prob = None

  def log_lik_gradient(self, x, y_next, infected):
    dim = len(x)
    if infected:
      inf_grad = gradient.logit_gradient(x, y_next, self.inf_params)
      not_inf_grad = np.zeros(dim)
    else:
      inf_grad = np.zeros(dim)
      not_inf_grad = gradient.logit_gradient(x, y_next, self.not_inf_params)
    return np.concatenate((inf_grad, not_inf_grad))

  def log_lik_hess(self, x, infected):
    dim = len(x)
    if infected:
      inf_hess = gradient.logit_hessian(x, self.inf_params)
      not_inf_hess = np.zeros((dim, dim))
    else:
      inf_hess = np.zeros((dim, dim))
      not_inf_hess = gradient.logit_hessian(x,  self.not_inf_params)
    return block_diag(inf_hess, not_inf_hess)

  def covariance(self, X, y, infected_locations):
    n, p = X.shape[1]
    grad_outer = np.zeros((2*p, 2*p))
    hess = np.zeros((2*p, 2*p))
    for i, x, y_ in enumerate(zip(X, y)):
      infected = i in infected_locations
      grad = self.log_lik_gradient(x, y_, infected)
      hess_i = self.log_lik_hess(x, infected)
      if infected:
        grad_outer[:p, :p] += np.outer(grad, grad)
        hess[:p, :p] += hess_i
      else:
        grad_outer[p:, p:] += np.outer(grad, grad)
        hess[p:, p:] += hess_i
    hess_inv = np.linalg.inv(hess + 0.1*np.eye(2*p))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(n)
    return cov

  def fit(self, X, y, weights, truncate, infected_locations, not_infected_locations):
    if is_y_all_1_or_0(y):
      y0 = y[0]
      n = len(y)
      expit_intercept_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
      intercept_ = logit(expit_intercept_)
      coef_ = np.zeros(X.shape[1])
      self.inf_params = self.not_inf_params = np.concatenate((intercept_, coef_))
    else:
      if weights is not None:
        inf_weights = weights[infected_locations]
        not_inf_weights = weights[not_infected_locations]
      else:
        inf_weights = not_inf_weights = None
      if len(infected_locations) > 0:
        if is_y_all_1_or_0(y[infected_locations]):
          inf_intercept_, inf_coef_ = empirical_bayes_coef(y[infected_locations], X.shape[1])
          inf_intercept_ = [inf_intercept_]
          self.inf_eb_prob = expit(inf_intercept_[0])
        else:
          self.reg_inf.fit(X[infected_locations], y[infected_locations])
          inf_intercept_, inf_coef_ = self.reg_inf.intercept_, self.reg_inf.coef_[0]
          self.inf_model_fitted = True
      if len(not_infected_locations) > 0:
        if is_y_all_1_or_0(y[not_infected_locations]):
          not_inf_intercept_, not_inf_coef_ = empirical_bayes_coef(y[not_infected_locations], X.shape[1])
          not_inf_intercept_ = [not_inf_intercept_]
          self.not_inf_eb_prob = expit(not_inf_intercept_[0])
        else:
          self.reg_not_inf.fit(X[not_infected_locations], y[not_infected_locations])
          not_inf_intercept_, not_inf_coef_ = self.reg_not_inf.intercept_, self.reg_not_inf.coef_[0]
          self.not_inf_model_fitted = True
      self.inf_params = np.concatenate((inf_intercept_, inf_coef_))
      self.not_inf_params = np.concatenate((not_inf_intercept_, not_inf_coef_))
    if truncate:
      cov = self.covariance(X, y, infected_locations)
      p = X.shape[1]
      new_params = np.random.multivariate_normal(np.concatenate((self.inf_params, self.not_inf_params)), cov=cov)
      self.inf_params = new_params[:p]
      self.not_inf_params = new_params[p:]

  def predict_proba(self, X, infected_locations, not_infected_locations):
    phat = np.zeros(X.shape[0])
    if len(phat[infected_locations]) > 0:
      if self.inf_model_fitted:
        phat[infected_locations] = self.reg_inf.predict_proba(X[infected_locations])[:, -1]
      else:
        phat[infected_locations] = self.inf_eb_prob
    if len(phat[not_infected_locations]) > 0:
      if self.not_inf_model_fitted:
        # phat[not_infected_locations] = self.reg_not_inf.predict_proba(X[not_infected_locations])[:, -1]
        logit_probs = np.dot(X[not_infected_locations, :], self.inf_params[1:]) + self.inf_params[0]
        phat[not_infected_locations] = expit(logit_probs)
      else:
        phat[not_infected_locations] = self.not_inf_eb_prob
    return phat

  @staticmethod
  def predict_proba_given_parameter(X, infected_locations, not_infected_locations, parameter):
    """

    :param X:
    :param infected_locations:
    :param not_infected_locations:
    :param parameter: array of the form [inf_intercept, inf_coef, not_inf_intercept, not_inf_coef]
    :return:
    """
    phat = np.zeros(X.shape[0])
    number_of_parameters = X.shape[1]

    # Get probabilities at infected locations
    inf_coef = parameter[1:number_of_parameters+1]
    inf_intercept = parameter[0]
    phat[infected_locations] = expit(np.dot(X[infected_locations, :], inf_coef) + inf_intercept)

    # Get probabilities at not-infected locations
    not_inf_coef = parameter[number_of_parameters+2:]
    not_inf_intercept = parameter[number_of_parameters+1]
    phat[not_infected_locations] = expit(np.dot(X[not_infected_locations, :], not_inf_coef) + not_inf_intercept)

    return phat


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
