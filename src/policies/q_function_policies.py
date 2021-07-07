import os
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, '..', 'estimation/q_functions/data')

from src.estimation.q_functions.fqi import fqi
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q, q_max_all_states
from src.estimation.model_based.sis.estimate_sis_q_fn import estimate_sis_q_fn
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
from src.estimation.model_based.Gravity.estimate_continuous_parameters import fit_continuous_grav_transition_model
from src.estimation.q_functions.model_fitters import SKLogit2
import src.estimation.q_functions.mse_optimal_combination as mse_combo
from src.estimation.q_functions.one_step import *
from src.estimation.q_functions.embedding import ggcn_multiple_runs, oracle_tune_ggcn, learn_ggcn
from src.environments.sis_infection_probs import sis_infection_probability_oracle_contaminated
from src.estimation.optim.quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
from src.utils.misc import random_argsort, kl
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.special import expit, logit
import numpy as np
import pickle as pkl
# import keras.backend as K
from functools import partial
import copy
import matplotlib.pyplot as plt
import torch


def two_step_true_probs_policy(**kwargs):
  N_FUNCTION_APPROX_REPS = 100
  N_REP = 10 

  env, evaluation_budget, treatment_budget, argmaxer, gamma = \
    kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], kwargs['gamma']

  # Define myopic oracle Q-function
  def oracle_qfn(a, s_, y_):
    infection_probs = env.infection_probability(a, y_, s_)
    return infection_probs

  # Get pseudo-outcomes
  def oracle_pseudo_outcome_exact(s_, y_):
    oracle_qfn_at_xraw = lambda a: oracle_qfn(a, s_, y_)
    a_ = argmaxer(oracle_qfn_at_xraw, evaluation_budget, treatment_budget, env)
    pseudo = oracle_qfn_at_xraw(a_)
    return pseudo

  # Estimate oracle pseudo outcome with function approximation
  s, y = env.current_state, env.current_infected
  s_indicator = (s > 0)
  a_dummy = np.concatenate((np.zeros(env.L - treatment_budget), np.ones(treatment_budget)))
  backups = []
  x_list = []
  for ix in range(N_FUNCTION_APPROX_REPS):
    print(ix)
    np.random.shuffle(a_dummy)
    infection_probs_at_a = env.infection_probability(a_dummy, y, s)
    y_next = np.random.binomial(n=1, p=infection_probs_at_a)
    s_next = env.update_state(s)
    backups.append(oracle_pseudo_outcome_exact(s_next, y_next))
    s_next_indicator = (s_next > 0)
    x_ = env.binary_state_psi(s_next_indicator, y_next)
    x_list.append(x_)

  _, oracle_pseudo_outcome_distilled = learn_ggcn(x_list, backups, env.adjacency_list, n_epoch=70,
                                                  target_are_probs=True,
                                                  samples_per_k=6, neighbor_subset_limit=2, verbose=False, lr=0.01,
                                                  batch_size=5, nhid=16, dropout=0.5, neighbor_order=1)

  def two_step_qfn(a):
    infection_probs_at_a = sis_infection_probability_oracle_contaminated(a, y, env.ETA, env.L, env.adjacency_list,
                                                                         env.epsilon, env.contaminator,
                                                                         env.binary_psi, **{'s': s, 'omega': 0})
    q_vals = np.zeros(env.L)
    for rep in range(N_REP):
      y_next = np.random.binomial(n=1, p=infection_probs_at_a)
      s_next = env.update_state(s)
      s_next_indicator = (s_next > 0)
      x_next = env.binary_state_psi(s_next_indicator, y_next)
      backup = oracle_pseudo_outcome_distilled(x_next)[:, 0]
      q_vals += (infection_probs_at_a + gamma * backup) / N_REP

    return q_vals

  a = argmaxer(two_step_qfn, evaluation_budget, treatment_budget, env)
  return a, {}


def two_step_ggcn_policy(**kwargs):
  env = kwargs['env']
  return two_step_ggcn_policy_helper(**kwargs)


def two_step_ggcn_policy_helper(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, raw_features, diagnostic_mode = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['raw_features'], kwargs['diagnostic_mode']

  GAMMA = 0.5
  ebola = hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX')

  if ebola:
    features0 = env.X
    features = env.X_raw
  else:
    features0 = env.X 
    features = env.X

  # Fit myopic q-function
  _, predictor0 = learn_ggcn(features0, env.y, env.adjacency_list, n_epoch=70, target_are_probs=False,
                             samples_per_k=6, neighbor_subset_limit=2, verbose=False, lr=0.005,
                             batch_size=5, nhid=16, dropout=0.0, neighbor_order=1)

  def qfn0(a, t):
      x_at_a = env.data_block_at_action(t, a)
      infection_probs = predictor0(x_at_a)
      return infection_probs

  # Get pseudo-outcomes
  def pseudo_outcome(t):
    qfn0_at_x = lambda a: qfn0(a, t)
    a_ = argmaxer(qfn0_at_x, evaluation_budget, treatment_budget, env)
    pseudo = qfn0_at_x(a_)
    return pseudo

  if diagnostic_mode:
    lm0 = LogisticRegression()
    lm0.fit(np.vstack(env.X_raw), np.hstack(env.y))
    def qfn0_baseline(a, x_raw):
      infection_probs = lm0.predict_proba(np.column_stack((a, x_raw[:, -1], x_raw[:, 0])))[:, 1]
      return infection_probs

    def pseudo_outcome_baseline(x_raw):
      qfn0_at_x_raw = lambda a: qfn0_baseline(a, x_raw)
      a_ = argmaxer(qfn0_at_x_raw, evaluation_budget, treatment_budget, env)
      pseudo = qfn0_at_x_raw(a_)
      return pseudo

    # Define myopic oracle Q-function
    def oracle_qfn(a, x_raw):
      infection_probs = env.infection_probability(a, x_raw[:, -1], x_raw[:, 0])
      return infection_probs

    # Get pseudo-outcomes
    def oracle_pseudo_outcome(x_raw):
      oracle_qfn_at_xraw = lambda a: oracle_qfn(a, x_raw)
      a_ = argmaxer(oracle_qfn_at_xraw, evaluation_budget, treatment_budget, env)
      pseudo = oracle_qfn_at_xraw(a_)
      return pseudo

  myopic = np.zeros(0)
  lookahead = np.zeros(0)
  backups = []
  myopic_baseline = np.zeros(0)
  lookahead_baseline = np.zeros(0)
  backups_baseline = []
  myopic_oracle = np.zeros(0)
  lookahead_oracle = np.zeros(0)
  backups_oracle = []

  for t, (x_raw_, x_raw_next_) in enumerate(zip(env.X_raw[:-1], env.X_raw[1:])):
    pseudo_rep = pseudo_outcome(t)
    myopic_rep = qfn0(x_raw_[:, 1], t)
    myopic = np.hstack((myopic, myopic_rep))
    lookahead = np.hstack((lookahead, pseudo_rep))
    backups.append(myopic_rep + GAMMA * pseudo_rep)

    if diagnostic_mode:
      myopic_baseline_rep = qfn0_baseline(x_raw_[:, 1], x_raw_)
      pseudo_baseline_rep = pseudo_outcome_baseline(x_raw_next_)
      myopic_baseline = np.hstack((myopic_baseline, myopic_baseline_rep))
      lookahead_baseline = np.hstack((lookahead_baseline, pseudo_baseline_rep))
      backup_baseline = myopic_baseline_rep + pseudo_baseline_rep
      backups_baseline.append(backup_baseline)

      myopic_oracle_rep = oracle_qfn(x_raw_[:, 1], x_raw_)
      pseudo_oracle_rep = oracle_pseudo_outcome(x_raw_next_)
      myopic_oracle = np.hstack((myopic_oracle, myopic_oracle_rep))
      lookahead_oracle = np.hstack((lookahead_oracle, pseudo_oracle_rep))
      backup_oracle = myopic_oracle_rep + pseudo_oracle_rep
      backups_oracle.append(backup_oracle)

  # Fit GGCN to backups to get q1
  current_x_raw = env.X_raw[-1]
  N_REP = 50
  dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
  eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)]
  # _, predictor = learn_ggcn(env.X[:-1], backups, env.adjacency_list, n_epoch=100, target_are_probs=True,
	# 		                      samples_per_k=15, neighbor_subset_limit=1, verbose=False, lr=0.01,
  #                           batch_size=10, nhid=16, dropout=0)

  _, predictor = learn_ggcn(features[:-1], backups, env.adjacency_list, n_epoch=100, target_are_probs=True,
                            samples_per_k=6, neighbor_subset_limit=2, verbose=False, lr=0.01,
                            batch_size=10, nhid=16, dropout=0.5, neighbor_order=1)

  if diagnostic_mode:
    _, predictor2 = learn_ggcn(features[:-1], backups_baseline, env.adjacency_list, n_epoch=100, target_are_probs=True,
                              samples_per_k=15, neighbor_subset_limit=2, verbose=False, lr=0.01,
                              batch_size=10, nhid=16, dropout=0.5)

    acc1_list = []
    acc2_list = []
    for x, b in zip(features[:-1], backups_oracle):
      backup_hat_1 = predictor(x)[:, 0]
      backup_hat_2 = predictor2(x)[:, 0]
      # error_1 = np.mean((backup_hat_1 - b1)**2)
      # error_2 = np.mean((backup_hat_2 - b2)**2)
      error_1 = np.mean((backup_hat_1 - b) ** 2)
      error_2 = np.mean((backup_hat_2 - b)**2)
      acc1_list.append(error_1)
      acc2_list.append(error_2)

    acc1 = np.mean(acc1_list)
    acc2 = np.mean(acc2_list)
    print(f'acc1: {acc1} acc2: {acc2}')

  # Get optimal action
  def qfn(a_):
    X_ = env.data_block_at_action(-1, a_, raw=ebola)
    return predictor(X_)[:, 0]

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, {}


def two_step_oracle_ggcn_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, raw_features = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['raw_features']

  # Define myopic oracle Q-function
  def oracle_qfn(a, x_raw):
      infection_probs = env.infection_probability(a, x_raw[:, -1], x_raw[:, 0])
      return infection_probs

  # Get pseudo-outcomes
  def oracle_pseudo_outcome(x_raw):
    oracle_qfn_at_xraw = lambda a: oracle_qfn(a, x_raw)
    a_ = argmaxer(oracle_qfn_at_xraw, evaluation_budget, treatment_budget, env)
    pseudo = oracle_qfn_at_xraw(a_)
    return pseudo

  backups = []
  for x_raw_, x_raw_next_ in zip(env.X_raw[:-1], env.X_raw[1:]):
    pseudo_x_raw = oracle_pseudo_outcome(x_raw_next_)
    myopic = oracle_qfn(x_raw_[:, 1], x_raw_)
    backup = myopic + pseudo_x_raw
    backups.append(backup)

  # Fit GGCN to backups to get q1
  # current_x_raw = env.X_raw[-1]
  # N_REP = 50
  # dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
  # eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)]
  # true_probs = np.hstack([oracle_qfn(a_, current_x_raw) for a_ in eval_actions])
  # _, predictor = learn_ggcn(env.X[:-1], backups, env.adjacency_list, n_epoch=100, target_are_probs=True,
	# 		                      samples_per_k=15, neighbor_subset_limit=1, verbose=False, lr=0.01,
  #                           batch_size=10, nhid=16, dropout=0)
  _, predictor = learn_ggcn(env.X[:-1], backups, env.adjacency_list, n_epoch=100, target_are_probs=True,
                            samples_per_k=15, neighbor_subset_limit=2, verbose=True, lr=0.01,
                            batch_size=10, nhid=16, dropout=0.5)
  # lm = Ridge()
  # lm.fit(np.vstack(env.X[:-1]), np.hstack(backups))
  # predictor2 = lambda x_: lm.predict(x_)

  # # Evaluate predictors (for diagnosis)
  # acc1_list = []
  # acc2_list = []
  # for x, b in zip(env.X[:-1], backups):
  #   backup_hat_1 = predictor(x)[:, 0]
  #   backup_hat_2 = predictor2(x)
  #   error_1 = np.mean((backup_hat_1 - b)**2)
  #   error_2 = np.mean((backup_hat_2 - b)**2)
  #   acc1_list.append(error_1)
  #   acc2_list.append(error_2)

  # acc1 = np.mean(acc1_list)
  # acc2 = np.mean(acc2_list)
  # print(f'acc1: {acc1} acc2: {acc2}')

  # Get optimal action
  def qfn(a_):
    X_ = env.data_block_at_action(-1, a_)
    return predictor(X_)[:, 0]

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, {}


def one_step_policy(**kwargs):
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, raw_features = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  def oracle_qfn(a):
      return env.next_infected_probabilities(a)

  if env.learn_embedding:
  # if env.learn_embedding:
    loss_dict = {}
    N_REP = 50
    dummy_act = np.concatenate((np.ones(treatment_budget), np.zeros(env.L - treatment_budget)))
    eval_actions = [np.random.permutation(dummy_act) for _ in range(N_REP)]
        
    if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
      X_raw = [np.column_stack((x_raw, env.NEIGHBOR_DISTANCE_MATRIX)) for x_raw in env.X_raw]
      env_name = 'Ebola'
    else:
      X_raw = env.X_raw
      env_name = 'sis'

    true_probs = np.hstack([oracle_qfn(a_) for a_ in eval_actions])


    # fname = os.path.join(data_dir, 'Ebola_data_t=8.p')
    # data_dict = {'X_raw_list': X_raw, 'y_list': env.y, 'adjacency_list': env.adjacency_list, 'eval_actions': eval_actions,
    #              'true_probs': true_probs, 'X_list': env.X, 'settings': {'env': env_name, 'L': env.L}}
    # pkl.dump(data_dict, open(fname, 'wb'))

    # _, predictor = learn_ggcn(X_raw, env.y, env.adjacency_list)

    # For diagnosis
    clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
    # clf = LogisticRegression()
    # clf.fit(np.vstack(X_raw), np.hstack(env.y))

    predictor, _ = oracle_tune_ggcn(env.X, env.y, env.adjacency_list, env, eval_actions, true_probs,
  		                    num_settings_to_try=1, n_epoch=300, neighbor_order=1)

    def qfn(a_):
      # X_raw_ = env.data_block_at_action(-1, a_, raw=True)
      X_ = env.data_block_at_action(-1, a_)
      # if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
      #   X_raw_ = np.column_stack((X_raw_, env.NEIGHBOR_DISTANCE_MATRIX))
      # return predictor(X_raw_)
      return predictor(X_)
    def linear_qfn(a_):
      return clf.predict_proba(env.data_block_at_action(-1, a_), **predict_proba_kwargs)
      # return clf.predict_proba(env.data_block_at_action(-1, a_, raw=True))[:, 1]

    
    def optimize_qfns(qfn_):
      a_ = argmaxer(qfn_, evaluation_budget, treatment_budget, env)
      q_a_true_ = oracle_qfn(a_).sum()
      return a_, q_a_true_

    a, q_a_true = optimize_qfns(qfn)
    a_linear = argmaxer(linear_qfn, evaluation_budget, treatment_budget, env)
    q_alin_true = oracle_qfn(a_linear).sum()
    q_a = qfn(a).sum()
    q_alin = qfn(a_linear).sum()

    # Get accuracy at actions at _this_ timestep
    linear_acc = 0.
    gccn_acc = 0.
    linear_diffs = np.zeros(0)
    gccn_diffs = np.zeros(0)
    for a_ in eval_actions:
      linear_probs = linear_qfn(a_)
      gccn_probs = qfn(a_)
      true_probs = oracle_qfn(a_)
      gccn_acc += kl(gccn_probs, true_probs) / N_REP
      linear_acc += kl(linear_probs, true_probs) / N_REP
      linear_diffs = np.concatenate((linear_diffs, np.abs(linear_probs - true_probs)))
      gccn_diffs = np.concatenate((gccn_diffs, np.abs(gccn_probs - true_probs)))

    print(f'gccn acc: {gccn_acc} linear acc: {linear_acc}\nq(a): {q_a} q(alin): {q_alin}\nq_true(a): {q_a_true} q_alin_true: {q_alin_true}')
    # plt.hist(linear_diffs, alpha=0.5, label='linear')
    # plt.hist(gccn_diffs, alpha=0.5, label='ggcn')
    # plt.legend()
    # plt.show()
    # pdb.set_trace()

    q_gccn_minus_q_linear = q_a_true - q_alin_true
    loss_dict['q_diff'] = q_gccn_minus_q_linear
  else:
    if raw_features:
      X = np.vstack(env.X_raw)
      y = np.hstack(env.y)
      clf = LogisticRegression()
      clf.fit(X, y)
      loss_dict = {}
      def qfn(a):
        return clf.predict_proba(env.data_block_at_action(-1, a, raw=True))[:, 1]
    else:
      clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
      # Add parameters to info dictionary if params is an attribute of clf (as is the case with SKLogit2)
      if 'params' in clf.__dict__.keys():
          loss_dict['q_fn_params'] = clf.params
      def qfn(a):
        return clf.predict_proba(env.data_block_at_action(-1, a), **predict_proba_kwargs)
    a = argmaxer(qfn, evaluation_budget, treatment_budget, env)

  return a, loss_dict


def one_step_projection_combo(**kwargs):
  """
  Combine one-step mf and mb estimators based on error of (linear) mf estimator and the projection of the mb
  estimator onto the space of linear models.

  :param kwargs:
  :return:
  """
  CANDIDATE_BANDWIDTHS = [0.01, 0.1, 1]
  KERNEL = lambda x, b: np.exp(-x**2 / b)

  env, evaluation_budget, treatment_budget, argmaxer, bootstrap, quantile = \
    kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['quantile']

  q_mb, q_mf, _, clf = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)

  # Project q_mb onto q_mf
  logit_q_model = Ridge()
  X_ref = clf.X_train
  logit_q_mb_target = np.array([])
  for x_raw in env.X_raw:
    # Clip q_mb
    q_mb_at_x = q_mb(x_raw)
    q_mb_at_x = np.maximum(np.minimum(q_mb_at_x, 0.99), 0.01)
    logit_q_mb_target = np.append(logit_q_mb_target, logit(q_mb_at_x))
  logit_q_model.fit(X_ref, logit_q_mb_target)

  # Select bandwidth by minimizing error on training set
  errors_at_bandwidths = np.zeros(len(CANDIDATE_BANDWIDTHS))
  projection_error = np.zeros(0)
  for t in range(env.T):
    x_raw = env.X_raw[t]
    x = env.X[t]
    x_times_infection = np.multiply(x, env.Y[t, :][:, np.newaxis])
    x_interaction = np.column_stack((x, x_times_infection))
    y = env.y[t]

    # Get error of projection
    logit_projection_prediction = logit_q_model.predict(x_interaction)
    projection_prediction = expit(logit_projection_prediction)
    infected_locations = np.where(x_raw[:, -1] == 1)[0]
    mf_prediction = q_mf(x, infected_locations, None)
    error = projection_prediction - mf_prediction
    mb_prediction = q_mb(x_raw)
    projection_error = np.append(projection_error, error)

    # Get error at each bandwidth
    for bandwidth_ix, bandwidth in enumerate(CANDIDATE_BANDWIDTHS):
      alpha = np.array([KERNEL(e_, bandwidth) for e_ in error])
      if not np.isfinite(alpha).all():
        alpha_not_nan = alpha[np.where(np.isfinite(alpha))]
        alpha[np.where(np.np.isfinite(alpha) != True)] = np.mean(alpha_not_nan)
      combined_prediction = alpha*mb_prediction + (1 - alpha)*mf_prediction
      loss = np.mean((combined_prediction - y)**2)
      errors_at_bandwidths[bandwidth_ix] += loss

  bandwidth = CANDIDATE_BANDWIDTHS[int(np.argmin(errors_at_bandwidths))]

  # Get mean of alphas for diagnostics
  alphas = np.array([KERNEL(e_, bandwidth) for e_ in projection_error])
  if not np.isfinite(alphas).all():
    alphas_not_nan = alphas[np.where(np.isfinite(alphas))]
    alphas[np.where(np.isfinite(alphas) != True)] = np.mean(alphas_not_nan)

  def qfn_combo(a):
    x_raw = env.data_block_at_action(-1, a, raw=True)
    x = env.data_block_at_action(-1, a)
    x_times_infection = np.multiply(x, x_raw[:, -1][:, np.newaxis])
    x_interaction = np.column_stack((x, x_times_infection))

    # Get error of projection
    logit_projection_prediction = logit_q_model.predict(x_interaction)
    projection_prediction = expit(logit_projection_prediction)
    infected_locations = np.where(x_raw[:, -1] == 1)[0]
    mf_prediction = q_mf(x, infected_locations, None)
    error = projection_prediction - mf_prediction

    # Combine
    alpha = np.array([KERNEL(e_, bandwidth) for e_ in error])
    if not np.isfinite(alpha).all():
      alpha_not_nan = alpha[np.where(np.isfinite(alpha))]
      alpha[np.where(np.isfinite(alpha) != True)] = np.mean(alpha_not_nan)

    mb_prediction = q_mb(x_raw)

    # if not np.isfinite(mb_prediction).all():
    #   pdb.set_trace()

    # if not np.isfinite(mf_prediction).all():
    #   pdb.set_trace()

    # if not np.isfinite(alpha).all():
    #   pdb.set_trace()

    return alpha*mb_prediction + (1-alpha)*mf_prediction

  a = argmaxer(qfn_combo, evaluation_budget, treatment_budget, env)
  return a, {'alpha_quantiles': np.percentile(alphas, [10, 50, 90])}


def one_step_truth_augmented(**kwargs):
  """
  Replace high-error probability estimates with true probabilities.

  :param kwargs:
  :return:
  """
  classifier, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, quantile = \
    kwargs['classifier'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap'], kwargs['quantile']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)
  bad_state_fname = os.path.join(this_dir, 'high-error-states.txt')
  high_error_states = np.loadtxt(bad_state_fname)

  def qfn(a):
    # Get absolute errors
    X_a = env.data_block_at_action(-1, a)
    phat = clf.predict_proba(X_a, **predict_proba_kwargs)
    true_probs = env.next_infected_probabilities(a, eta=env.ETA)
    # errors = np.abs(phat - true_probs)
    # outliers = np.where(errors > 0.1)

    high_error_locations = []
    for l, x in enumerate(X_a):
      for h in high_error_states:
          if np.array_equal(x[8:], h):
            high_error_locations.append(l)
            break

    # Replace outlier probabilities 
    phat[high_error_locations] = true_probs[high_error_locations]
    return phat

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)

  return a, loss_dict


def two_step_mb(**kwargs):
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  if env.__class__.__name__ == 'SIS':
    q_fn_at_block_, predict_proba_kwargs = fit_one_step_sis_mb_q(env, bootstrap_weights=weights)
  elif env.__class__.__name__ == 'Ebola':
    q_fn_at_block_, _ = fit_one_step_ebola_mb_q(env)

  def q_fn(raw_data_block, a):
    raw_data_block_at_action = np.column_stack((raw_data_block[:, 0], a, raw_data_block[:, 2]))
    return q_fn_at_block_(raw_data_block_at_action)

  # Back up once
  backup = []
  for t in range(1, env.T):
    q_fn_at_block_t = lambda a: q_fn_at_block_(np.column_stack((env.X_raw[t][:, 0], a, env.X_raw[t][:, 2])))
    a_max = argmaxer(q_fn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = q_fn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  y_infected = np.zeros(0) 
  y_not_infected = np.zeros(0)
  X_infected = np.zeros((0, env.X[0].shape[1]))
  X_not_infected = np.zeros((0, env.X[0].shape[1]))

  for t, x_ in enumerate(env.X[:-1]):
    infected_indices = np.where(env.X_raw[t][:, -1] == 1)
    not_infected_indices = np.where(env.X_raw[t][:, -1] == 0)
    y_infected = np.hstack((y_infected, env.y[t][infected_indices]))
    y_not_infected = np.hstack((y_not_infected, env.y[t][not_infected_indices]))
    X_infected = np.vstack((X_infected, x_[infected_indices]))
    X_not_infected = np.vstack((X_not_infected, x_[not_infected_indices]))

  reg_infected = regressor()
  reg_not_infected = regressor()

  reg_infected.fit(X_infected, y_infected)
  reg_not_infected.fit(X_not_infected, y_not_infected)

  X_infected = np.vstack([x_[np.where(env.X_raw[t][:, -1] == 1)] for t, x_ in enumerate(env.X[:-1])])
  X_not_infected = np.vstack([x_[np.where(env.X_raw[t][:, -1] == 0)] for t, x_ in enumerate(env.X[:-1])])

  def backedup_up_qfn(a):
    backup = np.zeros(env.L)
    phat = q_fn(env.X_raw[-1], a)
    infected_indices_ = np.where(env.X_raw[-1][:, -1] == 1)
    not_infected_indices_ = np.where(env.X_raw[-1][:, -1] == 0) 

    X_at_a = env.data_block_at_action(-1, a)
    backup[infected_indices] = reg_infected.predict(X_at_a[infected_indices])
    backup[not_infected_indices] = reg_not_infected.predict(X_at_a[not_infected_indices])

    return phat + gamma * backup

  a = argmaxer(backedup_up_qfn, evaluation_budget, treatment_budget, env)
  return a, None


def two_step(**kwargs):
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap, gamma, raw_features = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap'], kwargs['gamma'], kwargs['raw_features']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  if env.learn_embedding:
    _, predictor = learn_ggcn(env.X_raw, env.y, env.adjacency_list)

    # For diagnosis
    clf, predict_proba_kwargs, loss_dict = fit_one_step_predictor(classifier, env, weights)

    def qfn_at_block(block_index, a):
      return predictor(env.data_block_at_action(block_index, a, raw=True))
  else:
    if raw_features:
      clf = LogisticRegression()
      y = np.hstack(env.y)
      X_raw = np.vstack(env.X_raw)
      clf.fit(X_raw, y)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a, raw=True))[:, 1]
    else:
      clf, predict_proba_kwargs, info = fit_one_step_predictor(classifier, env, weights)
      def qfn_at_block(block_index, a):
        return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(1, env.T):
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  # reg = regressor(n_estimators=100)
  reg = regressor()
  if raw_features:
    reg.fit(np.vstack(env.X_raw[:-1]), np.hstack(backup))
  else:
    reg.fit(np.vstack(env.X[:-1]), np.hstack(backup))

  def qfn(a):
    infections = env.Y[-1, :]
    infected_indices = np.where(infections == 1)[0]
    not_infected_indices = np.where(infections == 0)[0]
    if raw_features:
      X0 = env.data_block_at_action(-1, a, raw=True)
      X1 = env.data_block_at_action(-1, a, raw=True)
      return clf.predict_proba(X0)[:, 1] + gamma * reg.predict(X1)
    else:
      X0 = env.data_block_at_action(-1, a)
      X1 = X0
      return clf.predict_proba(X0, infected_indices, not_infected_indices) + gamma * reg.predict(X1)

      
  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, {'q_fn_params': reg.coef_}


def two_step_stacked(**kwargs):
  MONTE_CARLO_REPS = 1  # For estimating model-based backup
  N_SPLITS = 10
  TRAIN_PROPORTION = 0.8

  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  # Train test-splits
  train_test_splits = []  # List of tuples (training_ixs, test_ixs), where each of these is a list of lists of indices
  for fold in range(N_SPLITS):
    training_ixs_for_fold = []
    test_ixs_for_fold = []
    for t in range(env.T):
      train_test_mask = np.random.binomial(1, TRAIN_PROPORTION, size=env.L)
      training_ixs_for_fold.append(np.where(train_test_mask == 1)[0])
      test_ixs_for_fold.append(np.where(train_test_mask == 0)[0])
    train_test_splits.append((training_ixs_for_fold, test_ixs_for_fold))

  # Stack to get q0 estimator
  # Fit models on training splits
  yhat_mb = np.zeros(0)
  yhat_mf = np.zeros(0)
  y = np.zeros(0)
  for fold in range(N_SPLITS):
    train_test_split = train_test_splits[fold]
    if env.__class__.__name__ == 'SIS':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])

    elif env.__class__.__name__ == 'Ebola':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])
    for t, (x_raw, x) in enumerate(zip(env.X_raw[:-1], env.X[:-1])):
      test_ixs = train_test_split[1][t]
      yhat_mb = np.append(yhat_mb, q_mb_fold(x_raw)[test_ixs])
      yhat_mf = np.append(yhat_mf, q_mf_fold(x[test_ixs, :], np.where(x_raw[test_ixs, -1] == 1)[0],
                                             np.where(x_raw[test_ixs, -1] == 0)))
      y = np.append(y, env.y[t][test_ixs])

  # Get optimal combination weight
  alpha_mb = np.sum(np.multiply(y - yhat_mf, yhat_mb - yhat_mf)) / np.linalg.norm(yhat_mb - yhat_mf)**2
  alpha_mb = np.min((1.0, np.max((0.0, alpha_mb))))

  # Stack q functions
  if env.__class__.__name__ == 'SIS':
    q_mb, q_mf, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)
  elif env.__class__.__name__ == 'Ebola':
    q_mb, q_mf, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2)

  def qfn0(a, t_):
    data_block = env.data_block_at_action(t_, a)
    raw_data_block = env.data_block_at_action(t_, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    q_mb_at_a = q_mb(raw_data_block)
    q_mf_at_a = q_mf(data_block, infected_indices[0], not_infected_indices[0])
    return alpha_mb * q_mb_at_a + (1 - alpha_mb) * q_mf_at_a

  # Stack to get v1(s,a) estimator (pseudo_outcome for short, though technically that's the whole backup)
  # Construct mf pseudo-outcomes
  pseudo_outcome_mf = np.zeros(0)
  for t in range(env.T-1):
    # Model-free pseudo-outcome
    a_tp1 = argmaxer(lambda a_: qfn0(a_, t+1), evaluation_budget, treatment_budget, env)
    pseudo_outcome_mf_t = qfn0(a_tp1, t+1)
    pseudo_outcome_mf = np.append(pseudo_outcome_mf, pseudo_outcome_mf_t)

  # Fit model to mf pseudo-outcomes
  reg = regressor()
  reg.fit(np.vstack(env.X_2[:-1]), pseudo_outcome_mf)

  def qfn1(a):
    print("Working")
    data_block = env.data_block_at_action(-1, a)
    data_block_2 = env.data_block_at_action(-1, a, neighbor_order=2)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)

    # Estimate q0
    q_mb_at_a = q_mb(raw_data_block)
    q_mf_at_a = q_mf(data_block, infected_indices[0], not_infected_indices[0])
    q0_at_a = alpha_mb*q_mb_at_a + (1-alpha_mb)*q_mf_at_a

    # Model-free prediction of v1
    q_mf_at_a = reg.predict(data_block_2)

    # Model-based prediction of v1
    q_mb_at_a = np.zeros(env.L)
    for mc_rep in range(MONTE_CARLO_REPS):

      def qfn0_mb(a):
        # Get prediction of next state
        y_draw = np.random.binomial(1, q0_at_a)
        raw_data_block_tp1 = np.column_stack((np.zeros(env.L), y_draw, a))
        X_tp1 = env.psi(raw_data_block_tp1, neighbor_order=1)
        infected_indices = np.where(y_draw == 1)

        # Get estimated q0 at predicted state
        q_mb_tp1 = q_mb(raw_data_block_tp1)
        q_mf_tp1 = q_mf(X_tp1, infected_indices[0], None)
        q_tp1 = alpha_mb*q_mb_tp1 + (1 - alpha_mb)*q_mf_tp1
        return q_tp1

      a_tp1 = argmaxer(qfn0_mb, evaluation_budget, treatment_budget, env)
      q_mb_at_a += qfn0_mb(a_tp1)
    q_mb_at_a /= MONTE_CARLO_REPS

    return alpha_mb*q_mb_at_a + (1 - alpha_mb)*q_mf_at_a

  a_ = argmaxer(qfn1, evaluation_budget, treatment_budget, env)
  return a_, {}


def two_step_higher_order(**kwargs):
  """
  Use second-order neighbor features rather than first order (as in two_step).
  :param kwargs:
  :return:
  """
  classifier, regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['classifier'], kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], \
    kwargs['argmaxer'], kwargs['bootstrap']

  if bootstrap:
    weights = np.random.exponential(size=len(env.X)*env.L)
  else:
    weights = None

  # One step
  clf, predict_proba_kwargs = fit_one_step_predictor(classifier, env, weights)
  def qfn_at_block(block_index, a):
    return clf.predict_proba(env.data_block_at_action(block_index, a), **predict_proba_kwargs)

  # Back up once
  backup = []
  for t in range(env.T-1):
    qfn_at_block_t = lambda a: qfn_at_block(t, a)
    a_max = argmaxer(qfn_at_block_t, evaluation_budget, treatment_budget, env)
    q_max = qfn_at_block_t(a_max)
    backup_at_t = env.y[t] + q_max
    backup.append(backup_at_t)

  # Fit backup-up q function
  reg = regressor()
  reg.fit(np.vstack(env.X_2[:-1]), np.hstack(backup))

  def qfn(a):
    return reg.predict(env.data_block_at_action(-1, a, neighbor_order=2))

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, None


def sis_mb_fqi(**kwargs):
  """
  Currently only two-step fqi!

  :param kwargs:
  :return:
  """
  env, treatment_budget, evaluation_budget, argmaxer, gamma, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['gamma'], kwargs['regressor']

  # Compute backup
  q_mb_one_step, _ = fit_one_step_sis_mb_q(env)
  mb_backup = mse_combo.sis_mb_backup(env, gamma, q_mb_one_step, q_mb_one_step, argmaxer, evaluation_budget,
                                      treatment_budget)

  # Fit q-function
  X_2 = np.vstack(env.X_2[:-1])
  reg = regressor()
  reg.fit(X_2, mb_backup)

  # Get optimal action
  def qfn(action):
    return q(action, -1, env, reg.predict, neighbor_order=2)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)

  return a, None


def sis_one_step_dyna(**kwargs):
  """
  Supplement rarely-visited states with synthetic data.

  :param kwargs:
  :return:
  """
  pass


def sis_model_based_policy(**kwargs):
  env, treatment_budget, evaluation_budget, argmaxer, planning_depth, bootstrap, \
    rollout_depth, gamma, classifier, regressor = \
    kwargs['env'], kwargs['treatment_budget'], kwargs['evaluation_budget'], kwargs['argmaxer'], \
    kwargs['planning_depth'], kwargs['bootstrap'], \
    kwargs['rollout_depth'], kwargs['gamma'], kwargs['classifier'], kwargs['regressor']

  auto_regressor = AutoRegressor(classifier, regressor)
  new_q_model = estimate_sis_q_fn(env, auto_regressor, rollout_depth, gamma, planning_depth,
                                  treatment_budget, evaluation_budget, argmaxer, bootstrap)

  q_hat = partial(q, data_block_ix=-1, env=env, predictive_model=new_q_model)
  a = argmaxer(q_hat, evaluation_budget, treatment_budget, env)
  K.clear_session()
  return a, new_q_model


def gravity_model_based_one_step(**kwargs):
  env, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  if env.__class__.__name__ == 'Ebola':
    eta, _ = fit_ebola_transition_model(env)
  elif env.__class__.__name__ == 'ContinuousGrav':
    eta = fit_continuous_grav_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  phat = one_step_q(a)
  ptrue = env.next_infected_probabilities(a)
  diff = np.abs(phat - ptrue)
  worst_ix = np.argmax(diff)
  print('eta: {}\ntrue eta: {}'.format(eta, env.ETA))
  # print('max loss: {} mean loss: {} worst ix: {}'.format(np.max(diff), np.mean(diff), worst_ix))
  return a, None


def sis_model_based_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  eta = fit_sis_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = np.zeros(env.L)
  probs = one_step_q(a)

  treat_ixs = random_argsort(probs, treatment_budget)
  a[treat_ixs] = 1
  return a, None


def continuous_model_based_myopic(**kwargs):
  env, treatment_budget = kwargs['env'], kwargs['treatment_budget']
  if env.__class__.__name__ == 'Ebola':
    eta = fit_ebola_transition_model(env)
  elif env.__class__.__name__ == 'ContinuousGrav':
    eta = fit_continuous_grav_transition_model(env)
  one_step_q = partial(env.next_infected_probabilities, eta=eta)
  a = np.zeros(env.L)
  probs = one_step_q(a)

  # Get priority score
  priorities_for_infected_locations = np.zeros(env.L)
  for l in np.where(env.current_infected == 1)[0]:
    for lprime in env.adjacency_list[l]:
      priorities_for_infected_locations[l] += \
        (1 - env.current_infected[lprime]) * probs[lprime] / env.distance_matrix[l, lprime]
  priorities_for_infected_locations /= np.sum(priorities_for_infected_locations)
  priority = probs

  # Treat greedily acc to priority
  priority[np.where(env.current_infected == 1)] = \
    priorities_for_infected_locations[np.where(env.current_infected == 1)]
  treat_ixs = random_argsort(-priority, treatment_budget)
  a[treat_ixs] = 1

  return a, None


def sis_model_based_one_step(**kwargs):
  env, bootstrap, argmaxer, evaluation_budget, treatment_budget = \
    kwargs['env'], kwargs['bootstrap'], kwargs['argmaxer'], kwargs['evaluation_budget'], kwargs['treatment_budget']
  eta = fit_sis_transition_model(env)
  print('eta hat: {} eta true: {}'.format(eta, env.ETA))
  one_step_q = partial(sis_infection_probability, y=env.current_infected, s=env.current_state, eta=eta,
                       omega=0, L=env.L, adjacency_lists=env.adjacency_list)
  a = argmaxer(one_step_q, evaluation_budget, treatment_budget, env)
  return a, None


def one_step_mse_averaged(**kwargs):
  env = kwargs['env']

  res = mse_combo.one_step_convex_combo(env)

  alpha_mb, alpha_mf, q_mb, q_mf = res['alpha_mb'], res['alpha_mf'], res['q_mb'], res['q_mf']

  # Get modified q_function
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    q_mb_at_a = q_mb(raw_data_block)
    q_mf_at_a = q_mf(data_block, infected_indices[0], not_infected_indices[0])
    q_combined = alpha_mb*q_mb_at_a + alpha_mf*q_mf_at_a
    return q_combined

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  # info = {'mb_bias': mb_bias, 'mb_var': mb_var, 'mf_var': mf_var, 'cov': mb_mf_cov, 'mf_bias': mf_bias}
  info = {}
  info.update({'alpha_mb': alpha_mb})
  return a, info


def one_step_stacked(**kwargs):
  N_SPLITS = 10
  TRAIN_PROPORTION = 0.8

  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  # Train-test splits
  train_test_splits = []  # List of tuples (training_ixs, test_ixs), where each of these is a list of lists of indices
  for fold in range(N_SPLITS):
    training_ixs_for_fold = []
    test_ixs_for_fold = []
    for t in range(env.T):
      train_test_mask = np.random.binomial(1, TRAIN_PROPORTION, size=env.L)
      training_ixs_for_fold.append(np.where(train_test_mask == 1)[0])
      test_ixs_for_fold.append(np.where(train_test_mask == 0)[0])
    train_test_splits.append((training_ixs_for_fold, test_ixs_for_fold))

  # Fit models on training splits
  yhat_mb = np.zeros(0)
  yhat_mf = np.zeros(0)
  y = np.zeros(0)
  for fold in range(N_SPLITS):
    train_test_split = train_test_splits[fold]
    if env.__class__.__name__ == 'SIS':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])

    elif env.__class__.__name__ == 'Ebola':
      q_mb_fold, q_mf_fold, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2, indices=train_test_splits[fold][0])
    for t, (x_raw, x) in enumerate(zip(env.X_raw[:-1], env.X[:-1])):
      test_ixs = train_test_split[1][t]
      yhat_mb = np.append(yhat_mb, q_mb_fold(x_raw)[test_ixs])
      yhat_mf = np.append(yhat_mf, q_mf_fold(x[test_ixs, :], np.where(x_raw[test_ixs, -1] == 1)[0],
                                             np.where(x_raw[test_ixs, -1] == 0)))
      y = np.append(y, env.y[t][test_ixs])

  # Get optimal combination weight
  alpha_mb = np.sum(np.multiply(y - yhat_mf, yhat_mb - yhat_mf)) / np.linalg.norm(yhat_mb - yhat_mf)**2
  alpha_mb = np.min((1.0, np.max((0.0, alpha_mb))))

  # Stack q functions
  if env.__class__.__name__ == 'SIS':
    q_mb, q_mf, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)
  elif env.__class__.__name__ == 'Ebola':
    q_mb, q_mf, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2)

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    q_mb_at_a = q_mb(raw_data_block)
    q_mf_at_a = q_mf(data_block, infected_indices[0], not_infected_indices[0])
    return alpha_mb * q_mb_at_a + (1 - alpha_mb) * q_mf_at_a

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  info = {}
  return a, info


def sis_one_step_equal_averaged(**kwargs):
  env = kwargs['env']
  
  if env.__class__.__name__ == 'SIS':
    q_mb, q_mf, _, _ = fit_one_step_sis_mf_and_mb_qs(env, SKLogit2)
  elif env.__class__.__name__ == 'Ebola':
    q_mb, q_mf, _, _ = fit_one_step_ebola_mf_and_mb_qs(env, SKLogit2)

  # Get modified q_function
  regressor, env, evaluation_budget, treatment_budget, argmaxer, bootstrap = \
    kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['treatment_budget'], kwargs['argmaxer'], \
    kwargs['bootstrap']

  def qfn(a):
    data_block = env.data_block_at_action(-1, a)
    raw_data_block = env.data_block_at_action(-1, a, raw=True)
    infected_indices, not_infected_indices = np.where(env.current_infected == 1), np.where(env.current_infected == 0)
    return 0.5 * q_mb(raw_data_block) + 0.5 * q_mf(data_block, infected_indices[0], not_infected_indices[0])

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  info = {}
  return a, info


def sis_two_step_mse_averaged(**kwargs):
  regressor, env, evaluation_budget, gamma, treatment_budget, argmaxer = \
      kwargs['regressor'], kwargs['env'], kwargs['evaluation_budget'], kwargs['gamma'], kwargs['treatment_budget'], \
      kwargs['argmaxer']

  # Get mse-averaged backup
  averaged_backup, info = mse_combo.two_step_sis_convex_combo(env, gamma, argmaxer, evaluation_budget, treatment_budget)

  # Fit q-function to backup
  X_2 = np.vstack(env.X_2[:-1])
  reg = regressor()
  reg.fit(X_2, averaged_backup)

  def qfn(action):
    return q(action, -1, env, reg.predict, neighbor_order=2)

  a = argmaxer(qfn, evaluation_budget, treatment_budget, env)
  return a, info
