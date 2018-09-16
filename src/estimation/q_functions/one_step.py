import pdb
import numpy as np
from src.estimation.model_based.sis import estimate_sis_parameters
from src.environments.sis_infection_probs import sis_infection_probability


def compare_with_true_probs(env, predictor, raw):
  if raw:
    phat = np.hstack([predictor(data_block) for data_block in env.X_raw])
  else:
      phat = np.hstack([predictor(data_block, np.where(raw_data_block[:, -1] == 1)[0], np.where(raw_data_block[:, -1] == 0)[0])
                        for raw_data_block, data_block in zip(env.X_raw, env.X)])
  true_expected_counts = np.hstack(env.true_infection_probs)
  max_loss = np.max(np.abs(phat - true_expected_counts))
  mean_loss = np.mean(np.abs(phat - true_expected_counts))
  print('mean loss {} max loss {}'.format(mean_loss, max_loss))
  return


def fit_one_step_predictor(classifier, env, weights, y_next=None, print_compare_with_true_probs=True):
  clf = classifier()
  if y_next is None:
    target = np.hstack(env.y).astype(float)
  else:
    target = y_next
  features = np.vstack(env.X)

  if clf.condition_on_infection:
    X_raw = np.vstack(env.X_raw)
    clf_kwargs = {'infected_locations': np.where(X_raw[:, -1] == 1),
                  'not_infected_locations': np.where(X_raw[:, -1] == 0)}
    predict_proba_kwargs = {'infected_locations': np.where(env.X_raw[-1][:, -1] == 1)[0],
                            'not_infected_locations': np.where(env.X_raw[-1][:, -1] == 0)[0]}
  else:
    clf_kwargs = {}
    predict_proba_kwargs = {}

  if weights is not None:
    weights = weights.flatten()
  clf.fit(features, target, weights, **clf_kwargs)
  if print_compare_with_true_probs:
    compare_with_true_probs(env, clf.predict_proba, False)
  return clf, predict_proba_kwargs


def fit_one_step_sis_mb_q(env, bootstrap_weights=None, y_next=None):
  # Get model-based
  eta = estimate_sis_parameters.fit_transition_model(env, bootstrap_weights=bootstrap_weights, y_next=y_next)

  def q_mb(data_block):
    infection_prob = sis_infection_probability(data_block[:, 1], data_block[:, 2], data_block[:, 0], eta, 0.0, env.L,
                                               env.adjacency_list)
    return infection_prob

  return q_mb, eta


def fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=None, y_next=None):
  """

  :param env:
  :param classifier:
  :param bootstrap_weights:
  :param y_next: If provided, fit to this rather than env.y.
  :return:
  """

  # Get model-based
  q_mb, mb_params = fit_one_step_sis_mb_q(env, bootstrap_weights=bootstrap_weights, y_next=y_next)

  # Get model-free
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, bootstrap_weights, y_next=y_next)

  def q_mf(data_block, infected_locations, not_infected_locations):
    return clf.predict_proba(data_block, infected_locations, not_infected_locations)

  # print('mb loss')
  # compare_with_true_probs(env, q_mb, raw=True)
  # print('mf loss')
  # compare_with_true_probs(env, q_mf, raw=False)

  return q_mb, q_mf, mb_params, clf


# def bootstrap_one_step_q_functions(env, classifier, B):
#   """
#   Return dict of 2 B-length lists of bootstrapped one-step model-based and model-free q functions and corresponding
#   list of B bootstrap weight arrays.
#   """
#
#   q_mf_list, q_mb_list, bootstrap_weight_list = [], [], []
#   for b in range(B):
#     bootstrap_weights = np.random.exponential(size=(env.T, env.L))
#     q_mb, q_mf, mb_params = fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=bootstrap_weights)
#     q_mf_list.append(q_mf)
#     q_mb_list.append(q_mb)
#     bootstrap_weight_list.append(bootstrap_weights)
#   return {'q_mf_list': q_mf_list, 'q_mb_list': q_mb_list, 'bootstrap_weight_list': bootstrap_weight_list}
#
#
# def bellman_error(env, q_fn, evaluation_budget, treatment_budget, argmaxer, gamma, use_raw_features=False):
#   r = np.hstack(np.array([np.sum(y) for y in env.y[1:]]))
#   if use_raw_features:
#     q = np.hstack(np.array([np.sum(q_fn(data_block)) for data_block in env.X_raw[:-1]]))
#   else:
#     q = np.hstack(np.array([np.sum(q_fn(data_block)) for data_block in env.X[:-1]]))
#   qp1_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, q_fn, argmaxer, raw=use_raw_features)
#   qp1_max = np.sum(qp1_max[1:, :], axis=1)
#   td = r + gamma * qp1_max - q
#   return np.linalg.norm(td)


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
