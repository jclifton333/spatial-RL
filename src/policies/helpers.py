from src.estimation.model_based.SIS.fit import fit_transition_model
from src.environments.sis_infection_probs import sis_infection_probability
import numpy as np
import pdb


def compare_with_true_probs(env, features, fitted_clf, clf_kwargs):
  true_expected_counts = np.hstack(env.true_infection_probs)
  phat = fitted_clf.predict_proba(features, **clf_kwargs)[:, -1]
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


def fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=None):
    # Get model-based
    eta = fit_transition_model(env, bootstrap_weights=bootstrap_weights)

    def q_mb(data_block):
      infection_prob = sis_infection_probability(data_block[:, 1], data_block[:, 2], data_block[:, 0], eta, 0.0, env.L,
                                                 env.adjacency_list)
      return infection_prob

    # Get model-free
    clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, bootstrap_weights)

    def q_mf(data_block):
      return clf.predict_proba(data_block, **predict_proba_kwargs)[:, -1]

    return q_mb, q_mf


def bootstrap_one_step_q_functions(env, classifier, B):
  """
  Return dict of 2 B-length lists of bootstrapped one-step model-based and model-free q functions and corresponding
  list of B bootstrap weight arrays.
  """

  q_mf_list, q_mb_list, bootstrap_weight_list = [], [], []
  for b in range(B):
    bootstrap_weights = np.random.exponential(size=(env.T, env.L))
    q_mb, q_mf = fit_one_step_mf_and_mb_qs(env, classifier, bootstrap_weights=bootstrap_weights)
    q_mf_list.append(q_mf)
    q_mb_list.append(q_mb)
    bootstrap_weight_list.append(bootstrap_weights)
  return {'q_mf_list': q_mf_list, 'q_mb_list': q_mb_list, 'bootstrap_weight_list': bootstrap_weight_list}