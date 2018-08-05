from src.estimation.model_based.SIS.fit import fit_transition_model
from src.estimation.q_functions.q_max import q_max_all_states
from src.environments.sis_infection_probs import sis_infection_probability
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

    print('mb loss')
    compare_with_true_probs(env, q_mb, raw=True)
    print('mf loss')
    compare_with_true_probs(env, q_mf, raw=False)

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


