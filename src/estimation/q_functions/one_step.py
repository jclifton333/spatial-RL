import pdb
import numpy as np
from src.estimation.model_based.sis import estimate_sis_parameters
from src.estimation.model_based.Ebola import estimate_ebola_parameters
from src.environments.sis_infection_probs import sis_infection_probability
from src.environments.gravity_infection_probs import ebola_infection_probs


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


def fit_one_step_predictor(classifier, env, weights, truncate=False, y_next=None, print_compare_with_true_probs=True,
                           indices=None):
  clf = classifier()
  if indices is None:
    if y_next is None:
      target = np.hstack(env.y).astype(float)
    else:
      target = y_next
    features = np.vstack(env.X)
  else:
    target = np.hstack([y_[ixs_at_t] for ixs_at_t, y_ in zip(indices, env.y)])
    features = np.vstack([x[ixs_at_t, :] for ixs_at_t, x in zip(indices, env.X)])

  if clf.condition_on_infection:
    if indices is None:
      X_raw = np.vstack(env.X_raw)
    else:
      X_raw = np.vstack([x_raw[ixs_at_t, :] for ixs_at_t, x_raw in zip(indices, env.X_raw)])
    clf_kwargs = {'infected_locations': np.where(X_raw[:, -1] == 1),
                  'not_infected_locations': np.where(X_raw[:, -1] == 0)}
    predict_proba_kwargs = {'infected_locations': np.where(env.X_raw[-1][:, -1] == 1)[0],
                            'not_infected_locations': np.where(env.X_raw[-1][:, -1] == 0)[0]}
  else:
    clf_kwargs = {}
    predict_proba_kwargs = {}

  if weights is not None:
    weights = weights.flatten()
  clf.fit(features, target, weights, truncate, **clf_kwargs)

  if print_compare_with_true_probs:
    compare_with_true_probs(env, clf.predict_proba, False)
  return clf, predict_proba_kwargs


def fit_one_step_sis_mb_q(env, bootstrap_weights=None, y_next=None, indices=None):
  # Get model-based
  eta = estimate_sis_parameters.fit_sis_transition_model(env, bootstrap_weights=bootstrap_weights, y_next=y_next,
                                                         indices=indices)

  def q_mb(data_block):
    infection_prob = sis_infection_probability(data_block[:, 1], data_block[:, 2], eta, env.L, env.adjacency_list,
                                               **{'s': data_block[:, 0], 'omega': 0.0})
    return infection_prob

  return q_mb, eta


def fit_one_step_ebola_mb_q(env, y_next=None, indices=None):
  # Get model-based
  eta = estimate_ebola_parameters.fit_ebola_transition_model(env, y_next=y_next, indices=indices)

  def q_mb(data_block):
    infection_prob = ebola_infection_probs(data_block[:, 1], data_block[:, 2], eta, env.L, env.adjacency_list,
                                           **{'distance_matrix': env.DISTANCE_MATRIX,
                                              'susceptibility': env.SUSCEPTIBILITY})
    return infection_prob

  return q_mb, eta


def fit_one_step_sis_mf_and_mb_qs(env, classifier, bootstrap_weights=None, y_next=None, indices=None):
  """
  :param env:
  :param classifier:
  :param bootstrap_weights:
  :param y_next: If provided, fit to this rather than env.y.
  :param indices: list of lists of location indices for train/test split, or None
  :return:
  """

  # Get model-based
  q_mb, mb_params = fit_one_step_sis_mb_q(env, bootstrap_weights=bootstrap_weights, y_next=y_next, indices=indices)

  # Get model-free
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, bootstrap_weights, y_next=y_next, indices=indices)

  def q_mf(data_block, infected_locations, not_infected_locations):
    return clf.predict_proba(data_block, infected_locations, not_infected_locations)

  # print('mb loss')
  # compare_with_true_probs(env, q_mb, raw=True)
  # print('mf loss')
  # compare_with_true_probs(env, q_mf, raw=False)

  return q_mb, q_mf, mb_params, clf


def fit_one_step_ebola_mf_and_mb_qs(env, classifier, bootstrap_weights=None, y_next=None, indices=None):
  """

  :param env:
  :param classifier:
  :param bootstrap_weights:
  :param y_next: If provided, fit to this rather than env.y.
  :param indices: list of lists of location indices for train/test split, or None
  :return:
  """

  # Get model-based
  q_mb, mb_params = fit_one_step_ebola_mb_q(env, y_next=y_next, indices=indices)

  # Get model-free
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, bootstrap_weights, y_next=y_next, indices=indices)

  def q_mf(data_block, infected_locations, not_infected_locations):
    return clf.predict_proba(data_block, infected_locations, not_infected_locations)

  # print('mb loss')
  # compare_with_true_probs(env, q_mb, raw=True)
  # print('mf loss')
  # compare_with_true_probs(env, q_mf, raw=False)

  return q_mb, q_mf, mb_params, clf

