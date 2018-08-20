from src.estimation.model_based.SIS.fit import fit_transition_model
from src.estimation.q_functions.q_max import q_max_all_states
from src.environments.sis_infection_probs import sis_infection_probability
from sklearn.linear_model import LogisticRegression
import numpy as np
import pdb


def compare_with_true_probs(env, predictor, raw):
  if raw:
    phat = np.hstack([predictor(data_block ) for data_block in env.X_raw])
  else:
    phat = np.hstack([predictor(data_block) for data_block in env.X])
  true_expected_counts = np.hstack(env.true_infection_probs)
  loss = np.mean((phat - true_expected_counts) ** 2)
  print('loss {}'.format(loss))
  return


def fit_one_step_predictor(classifier, env, weights, print_compare_with_true_probs=False):
  clf = classifier()
  target = np.hstack(env.y).astype(float)
  features = np.vstack(env.X)

  if clf.condition_on_infection:
    X_raw = np.vstack(env.X_raw)
    clf_kwargs = {'infected_locations': np.where(X_raw[:, -1] == 1),
                  'not_infected_locations': np.where(X_raw[:, -1] == 0)}
    predict_proba_kwargs = {'infected_locations': np.where(env.X_raw[-1][:, -1] == 1),
                            'not_infected_locations': np.where(env.X_raw[-1][:, -1] == 0)}
  else:
    clf_kwargs = {}
    predict_proba_kwargs = {}

  if print_compare_with_true_probs:
    compare_with_true_probs(env, features, clf, clf_kwargs)

  if weights is not None:
    weights = weights.flatten()
  clf.fit(features, target, weights, **clf_kwargs)
  return clf, predict_proba_kwargs


def fit_one_step_mb_q(env, bootstrap_weights=None):
  # Get model-based
  eta = fit_transition_model(env, bootstrap_weights=bootstrap_weights)

  def q_mb(data_block):
    infection_prob = sis_infection_probability(data_block[:, 1], data_block[:, 2], data_block[:, 0], eta, 0.0, env.L,
                                               env.adjacency_list)
    return infection_prob

  return q_mb, eta


def fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=None):
  # Get model-based
  q_mb, mb_params = fit_one_step_mb_q(env, bootstrap_weights=bootstrap_weights)

  # Get model-free
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, bootstrap_weights)

  def q_mf(data_block):
    return clf.predict_proba(data_block, **predict_proba_kwargs)[:, -1]

  print('mb loss')
  compare_with_true_probs(env, q_mb, raw=True)
  print('mf loss')
  compare_with_true_probs(env, q_mf, raw=False)

  return q_mb, q_mf, mb_params, clf


def bootstrap_one_step_q_functions(env, classifier, B):
  """
  Return dict of 2 B-length lists of bootstrapped one-step model-based and model-free q functions and corresponding
  list of B bootstrap weight arrays.
  """

  q_mf_list, q_mb_list, bootstrap_weight_list = [], [], []
  for b in range(B):
    bootstrap_weights = np.random.exponential(size=(env.T, env.L))
    q_mb, q_mf, mb_params = fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=bootstrap_weights)
    q_mf_list.append(q_mf)
    q_mb_list.append(q_mb)
    bootstrap_weight_list.append(bootstrap_weights)
  return {'q_mf_list': q_mf_list, 'q_mb_list': q_mb_list, 'bootstrap_weight_list': bootstrap_weight_list}


def bellman_error(env, q_fn, evaluation_budget, treatment_budget, argmaxer, gamma, use_raw_features=False):
  r = np.hstack(np.array([np.sum(y) for y in env.y[1:]]))
  if use_raw_features:
    q = np.hstack(np.array([np.sum(q_fn(data_block)) for data_block in env.X_raw[:-1]]))
  else:
    q = np.hstack(np.array([np.sum(q_fn(data_block)) for data_block in env.X[:-1]]))
  qp1_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, q_fn, argmaxer, raw=use_raw_features)
  qp1_max = np.sum(qp1_max[1:, :], axis=1)
  td = r + gamma * qp1_max - q
  return np.linalg.norm(td)


def estimate_mb_bias_and_variance(phat, env):
  """

  :param phat: Estimated one step probabilities
  :param env:
  :return:
  """
  NUMBER_OF_BOOTSTRAP_SAMPLES = 100
  THRESHOLD = 0.05  # For softhresholding bias estimate

  mb_bias = 0.0
  phat_mean_replicates = []
  for b in range(NUMBER_OF_BOOTSTRAP_SAMPLES):
    one_step_q_b = fit_one_step_mb_q(env, np.random.multinomial(env.L, np.ones(env.L)/env.L, size=env.T))
    phat_b = np.hstack([one_step_q_b(data_block) for data_block in env.X])
    mb_bias += (np.mean(phat - phat_b) - mb_bias) / (b + 1)
    phat_mean_replicates.append(np.mean(phat_b))
  # mb_variance = np.var(phat_mean_replicates)

  mb_variance = estimate_mb_variance(phat, env)

  print('bias before threshold: {}'.format(mb_bias))
  mb_bias = softhresholder(mb_bias, THRESHOLD)
  return mb_bias, mb_variance


def estimate_mb_variance(phat, env):
  # Get mb variance from asymptotic normality
  X_raw = np.vstack(env.X_raw)
  V = np.diag(np.multiply(phat, 1 - phat))
  beta_hat_cov_inv = np.dot(X_raw.T, np.dot(V, X_raw))
  beta_hat_cov = np.linalg.inv(beta_hat_cov_inv + 0.01*np.eye(beta_hat_cov_inv.shape[0]))
  X_beta_hat_cov = np.dot(X_raw, np.dot(beta_hat_cov, X_raw.T))
  mb_variance = np.sum(np.multiply(np.diag(X_beta_hat_cov), expit_derivative(phat)**2)) / len(phat)**2
  return mb_variance


# def estimate_cov_and_mb_bias(phat, env, classifier):
#   """
#
#   :param phat:
#   :param env:
#   :param classifier: This should be a flexible model.
#   :return:
#   """
#   NUMBER_OF_BOOTSTRAP_SAMPLES = 100
#
#   clf = classifier()
#   X, y = np.vstack(env.X), np.hstack(env.y).astype(float)
#   clf.fit(X, y)
#   phat_parametric = clf.predict_proba(X)[:, -1]
#   phat_parametric_mean = np.mean(phat)
#   correlation = 0.0
#
#   # Bias estimate
#   mb_bias = phat - phat_parametric
#
#   # Parametric bootstrap correlation estimate
#   # for b in range(NUMBER_OF_BOOTSTRAP_SAMPLES):
#   #   y_b = np.random.binomial(1, p=phat_parametric)
#   #   clf_b = classifier()
#   #   clf_b.fit(X, y_b)
#  #   phat_b = clf.predict_proba(X)
#   #   correlation += ((np.mean(phat_b) - phat_parametric_mean)*(np.mean(y_b) - phat_parametric_mean)
#   #                   - correlation) / (b + 1)
#
#   return softhresholder(np.mean(mb_bias), 0.05), 0.0


def estimate_mf_variance(phat):
  """
  Parametric estimate of one step mf variance, using one step mb.
  :param phat: estimated one step probabilities
  :param env:
  """
  mf_variance = np.multiply(phat, 1 - phat)
  return np.sum(mf_variance) / len(mf_variance)**2


def estimate_alpha_from_mse_components(q_mb, mb_bias, mb_variance, mf_variance, correlation):
  """
  Get estimated mse-optimal convex combination weight for combining q-functions, ignorning correlations between
  estimators.
  :param q_mb:
  :param mb_bias:
  :param mb_variance:
  :param mf_variance:
  :return:
  """
  q_mb_mean = np.mean(q_mb)
  preliminary_backup_estimate = q_mb_mean - mb_bias

  # Estimate bias coefficients k
  k_mb = q_mb_mean / preliminary_backup_estimate
  k_mf = 1

  # Estimate coefficients of variation v
  v_mf = mf_variance / preliminary_backup_estimate**2
  v_mb = mb_variance / preliminary_backup_estimate**2

  # lambda_ = np.sqrt((k_mf**2 * v_mb) / (k_mb**2 * v_mf))
  # correlation = 0.0

  # Estimate alpha
  alpha_mf = (mb_variance + mb_bias**2 - correlation) / (mb_bias**2 + mb_variance + mf_variance - 2*correlation)

  # Clip to [0, 1]
  alpha_mf = np.max((0.0, np.min((1.0, alpha_mf))))
  alpha_mb = 1 - alpha_mf
  return alpha_mb, alpha_mf


def estimate_mse_optimal_convex_combination(q_mb_one_step, env):
  phat = np.hstack([q_mb_one_step(data_block) for data_block in env.X_raw])
  mb_bias, mb_variance = estimate_mb_bias_and_variance(phat, env)
  covariance = 0.0
  mf_variance = estimate_mf_variance(phat)
  mb_variance = np.min((mb_variance, mf_variance))  # We know that mb_variance must be smaller than mf_variance
  alpha_mb, alpha_mf = estimate_alpha_from_mse_components(phat, mb_bias, mb_variance, mf_variance, covariance)
  mse_components = {'mb_bias': mb_bias, 'mb_variance': mb_variance, 'mf_variance': mf_variance,
                    'covariance': covariance}
  return alpha_mb, alpha_mf, phat, mse_components


def expit_derivative(x):
  """
  For delta method estimate of mb variance.
  :param x:
  :return:
  """
  return np.exp(-x) / (1 + np.exp(-x))**2


def softhresholder(x, threshold):
  if -threshold < x < threshold:
    return 0
  else:
    return x - np.sign(x)*threshold






