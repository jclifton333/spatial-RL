import pdb
import numpy as np
import src.utils.gradient as gradient
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from scipy.linalg import block_diag
from scipy.special import expit, logit
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add
from keras.regularizers import L1L2
from keras import backend as K
import tensorflow as tf
from keras import optimizers
import talos as ta


def keras_hyperparameter_search(X, y, model_name, best_params=None, clf=False, test=False):
    """

    :param X:
    :param y:
    :param clf: If clf assume y's binary and use CE loss, otherwise use SE loss.
    :param test: don't try many combos if just testing
    :return:
    """
    # Following https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53
    params = {
      'epochs': [1, 20, 50],
      'units1': [20, 50, 100],
      'lr': (0.5, 5, 5)
    }
    input_shape = X.shape[1]

    # Define model as function of grid params
    def model(X_train, y_train, X_val, y_val, params):
      reg = Sequential()
      reg.add(Dense(units=int(params['units1']), input_dim=input_shape, activation='relu'))
      reg.add(Dense(1, activation='sigmoid'))
      if clf:
        loss = 'binary_crossentropy'
      else:
        loss = 'mean_squared_error'
      reg.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
      if X_val is not None:
        history = reg.fit(X_train, y_train, verbose=True, epochs=int(params['epochs']),
                          validation_data=[X_val, y_val])
      else:
        history = reg.fit(X_train, y_train, verbose=True, epochs=int(params['epochs']))
      return history, reg

    # Search
    if test:
      proportion_to_sample = 0.05
    else:
      proportion_to_sample = 0.1
    search = ta.Scan(x=X, y=y, model=model, dataset_name=model_name, grid_downsample=proportion_to_sample,
                     params=params)

    # # Get best model
    # best_params = ta.Reporting(search).table().sort_values(by='val_acc', ascending=False).iloc[0]
    # best_params = {
    #   'units1': int(best_params['units1']),
    #   'dropout1': float(best_params['dropout1']),
    #   'units2': int(best_params['units2']),
    #   'dropout2': float(best_params['dropout2']),
    #   'lr': float(best_params['lr']),
    #   'epochs': int(best_params['epochs'])
    # }

    # if best_params is None:
    #   best_params = {
    #     'units1': 200,
    #     'dropout1': 0.0,
    #     'lr': 0.001,
    #     'epochs': 20
    #   }

    # graph = tf.Graph()
    # with graph.as_default():
    #   session = tf.Session()
    #   init = tf.global_variables_initializer()
    #   session.run(init)
    #   with session.as_default():
    # _, best_reg = model(X, y, None, None, best_params)
    # predictor = best_reg

    predictor = ta.Predict(search)

    return predictor


def fit_piecewise_keras_classifier(X, y, model_name, best_params=None,
                                   test=False, tune=True):
  """
  FIt separate kears infection probability models.
  :param X:
  :param y:
  :return:

  """
  input_shape = X.shape[1]

  if not tune:
    reg = Sequential()
    reg.add(Dense(units=50, input_dim=input_shape, activation='relu'))
    reg.add(Dense(units=50, activation='relu'))
    reg.add(Dense(1, activation='sigmoid'))
    reg.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    reg.fit(X, y, sample_weight=None, verbose=True, epochs=5)

  else:
    reg = keras_hyperparameter_search(X, y, model_name + '-infected',
                                      best_params=best_params, clf=True, test=test)

  def predict_proba_piecewise(X_):
    probs = reg.predict(X_)
    return probs

  # return reg, graph
  return predict_proba_piecewise


def fit_piecewise_keras_regressor(X, y, model_name, tune=True, test=False):
  """
  Fit separate regression models for infected and not-infected locations.

  :param X:
  :param y:
  :param infected_indices:
  :param not_infected_indices:
  :return:
  """
  if not tune:
    params = {
        'dropout1': 0.17,
        'dropout2': 0.17,
        'epochs': 20,
        'units1': 20,
        'units2': 10,
        'lr': 1.4
    }
    input_shape = X.shape[1]

    reg = Sequential()
    reg.add(Dense(params['units1'], input_dim=input_shape, activation='relu', kernel_initializer='normal'))
    reg.add(Dropout(params['dropout1']))
    reg.add(Dense(params['units2'], activation='relu', kernel_initializer='normal'))
    reg.add(Dropout(params['dropout2']))
    reg.add(Dense(1))
    reg.compile(optimizer='adam', loss='mean_squared_error')
    reg.fit(X, y, verbose=True, epochs=params['epochs'])

  else:
    reg = keras_hyperparameter_search(X, y, model_name + 'infected', clf=False,
                                      test=test)

  # def predict_piecewise(X_):
  #   predictions = reg.predict(X_).flatten()
  #   return predictions

  # return predict_piecewise
  return reg


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
      return self.predictor.predict(X).reshape(-1)
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
    self.negative_log_likelihood = None

  def fit(self, X, y, weights):
    if is_y_all_1_or_0(y):
      self.intercept_, self.coef_ = empirical_bayes_coef(y, X.shape[1])
    else:
      self.reg.fit(X, y, sample_weight=weights)
      self.get_coef()

      # Negative log likelihood
      phat = self.reg.predict_proba(X)[:, -1]
      log_likelihood_elements = y * np.log(phat) + (1 - y) * np.log(1 - phat)
      self.negative_log_likelihood = -np.sum(log_likelihood_elements)

  def get_coef(self):
    self.intercept_ = self.reg.intercept_
    self.coef_ = self.reg.coef_

  def predict_proba(self, X):
    phat = self.reg.predict_proba(X)
    return phat[:, -1]


class SKLogit2(object):
  condition_on_infection = True

  def __init__(self):
    self.reg_= LogisticRegression()
    # self.reg_ = MLPClassifier(hidden_layer_sizes=(50,50))
    self.model_fitted = False
    self.params = None
    self.eb_prob = None
    self.aic = None

  def log_lik_gradient(self, x, y_next, infected):
    x_inf = infected * x
    x_interaction = np.concatenate(([1], x, x_inf))
    grad = gradient.logit_gradient(x_interaction, y_next, self.params)
    return grad

  def log_lik_hess(self, x, infected):
    x_inf = infected * x
    x_interaction = np.concatenate(([1], x, x_inf))
    hess = gradient.logit_hessian(x_interaction, self.params)
    return hess

  def covariance(self, X, y, infected_locations):
    n, p = X.shape[1]
    grad_outer = np.zeros((2*p, 2*p))
    hess = np.zeros((2*p, 2*p))
    for i, x, y_ in enumerate(zip(X, y)):
      infected = i in infected_locations
      grad = self.log_lik_gradient(x, y_, infected)
      hess_i = self.log_lik_hess(x, infected)
      grad_outer += np.outer(grad, grad)
      hess += hess_i
    hess_inv = np.linalg.inv(hess + 0.1*np.eye(2*p))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(n)
    return cov

  def fit(self, X, y, weights, truncate, infected_locations, not_infected_locations):
    if is_y_all_1_or_0(y):
      y0 = y[0]
      n = len(y)
      expit_intercept_ = (1 + y0*n) / (2 + n)  # Smoothed estimate
      intercept_ = logit(expit_intercept_)
      coef_ = np.zeros(X.shape[1]*2)
      self.params = np.concatenate((intercept_, coef_))
      self.eb_prob = expit(intercept_[0])
    else:
      infection_indicator = np.array([i in infected_locations[0] for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((X, X_times_infection))
      self.X_train = X_interaction
      self.reg_.fit(X_interaction, y)
      self.model_fitted = True
      self.params = np.concatenate(([self.reg_.intercept_, self.reg_.coef_[0]]))
    if truncate:  # ToDo: modify to reflect not-split model
      cov = self.covariance(X, y, infected_locations)
      p = X.shape[1]
      new_params = np.random.multivariate_normal(np.concatenate((self.inf_params, self.not_inf_params)), cov=cov)
      self.inf_params = new_params[:p]
      self.not_inf_params = new_params[p:]

    # Negative log likelihood
    phat = self.reg_.predict_proba(X_interaction)[:, -1]
    self.log_likelihood_elements = y * np.log(phat) + (1 - y) * np.log(1 - phat)
    negative_log_likelihood = -np.sum(self.log_likelihood_elements)
    n, p = X_interaction.shape
    # self.aic = p + negative_log_likelihood + (p**2 + p) / np.max((1.0, n - p - 1))  # Technically, AICc/2

  def predict_proba(self, X, infected_locations, not_infected_locations):
    if self.model_fitted:
      infection_indicator = np.array([i in infected_locations for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((X, X_times_infection))
      phat = self.reg_.predict_proba(X_interaction)[:, -1]
    else:
      phat = self.eb_prob
    return phat

  def predict_proba_given_parameter(self, X, infected_locations, not_infected_locations, parameter):
    """
    :param X:
    :param infected_locations:
    :param not_infected_locations:
    :param parameter: array of the form [inf_intercept, inf_coef, not_inf_intercept, not_inf_coef]
    :return:
    """
    if self.model_fitted:
      infection_indicator = np.array([i in infected_locations for i in range(X.shape[0])])
      X_times_infection = np.multiply(X, infection_indicator[:, np.newaxis])
      X_interaction = np.column_stack((np.ones(X.shape[0]), X, X_times_infection))
      logit_phat = np.dot(X_interaction, parameter)
      phat = expit(logit_phat)
    else:
      phat = self.eb_prob
    return phat

