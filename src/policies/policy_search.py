"""
From RSS paper, pdf pg. 13.

Decision rule construction (suppressing dependnce on time t)
  Can treat c locations
  E is space of parameters \eta
  R(s, a; eta) is a vector of priority scores (one per location), assuming locations j where
    a_j = 1 are going to be treated
  for nonnegative integers m, define
    U_l(s, a; eta, m) = 1 if R_l(s, a; eta) >= R_(m)(s, a; eta) ( i.e. mth order stat)
                        0 o.w.
  take k <= c
  define d^(1)(s; eta) to be the binary vector that selects floor(c / k) highest-priority locs
    Let w^(1) denote d^(1)(s; eta)
  Recursively, for j=2,...,k
    w^(j) = d^(j)(s; eta)
    delta_j = floor(j*c / k) - floor((j-1)*c/k)
    d^(j) = U(s, w^(j-1); eta, delta_j) + w^(j-1)
  Final decision rule is d^(k)(s; eta)

  C^T(d; beta, theta) is expected cumulative value under decision rule d, against model parametrized by theta, beta,
  up to horizon T.

  Need to take argmax of C^T(d; \hat{beta}, \hat{theta}) over class of policies described above.

Inputs
  T
  S
  eta_0
  f(y | s, a, beta)
  g(s | s, a, theta)
  { alpha_j }_j>=1
  { zeta_j }_j>=1
  tol > 0

set k = 1, \tilde{S} = S
draw Z^k from unif(-1, 1)^d

while alpha_k >= tol
  from m = 1, ..., T-1
    set A^t+m = d(S^t+m, eta^k + zeta_k Z^k)
    draw S^t+m+1 ~ g(s^t+m+1 | s^t+m, a^t+m; theta)
    draw Y^t+m ~ f(y^t+m | s^t+m, a^t+m; beta)
    set \tilde{A}^t+m
    draw \tilde{Y}^t+m...
    draw \tilde{S}^t+m+1...
  set eta^k+1 = G_E [ eta^k + alpha_k / (2 zeta_k) (Z^k 1^T_L (Y^t+T-1 - \tilde{Y}^t+T-1))]
  set k = k + 1
output eta_k

where G_E(x) is the projection of x onto the parameter space E (where eta lives)
"""
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)

import pdb
import numpy as np
import copy

import src.environments.sis_infection_probs as sis_inf_probs
from src.estimation.model_based.sis.estimate_sis_parameters import fit_infection_prob_model


def R(env, s, a, y, infection_probs_predictor, transmission_prob_predictor, data_depth, eta, beta):
  """
  Linear priority score function.

  """
  priority_features = features_for_priority_score(env, s, a, y, infection_probs_predictor, transmission_prob_predictor,
                                                  data_depth, beta)
  return np.dot(priority_features, eta)


def update_eta(eta, alpha, zeta, z, y, y_tilde):
  ones = np.ones(len(y))
  second_term = z * np.dot(ones, y - y_tilde)
  new_eta = eta + alpha / (2 * zeta) * second_term
  new_eta_norm = np.linalg.norm(new_eta)
  new_eta /= np.max((1.0, new_eta_norm))  # Project onto unit sphere.
  return new_eta


def U(priority_scores, m):
  """

  :param priority_scores: Corresponds to R above.
  :param m: Integer >= 1.
  :return:
  """
  priority_scores_mth_order_stat = np.argsort(priority_scores)[int(m)]  # ToDo: Optimize?
  U = priority_scores >= priority_scores_mth_order_stat
  return U


def decision_rule(env, s, a, y, infection_probs_predictor, transmission_probs_predictor, eta, beta, k, treatment_budget,
                  priority_scores):

  d = np.zeros(len(priority_scores))
  floor_c_by_k = int(np.floor(treatment_budget / k))
  d[np.argsort(-priority_scores)][:floor_c_by_k] = 1
  for j in range(1, k):
    w = d
    delta_j = np.floor(j * treatment_budget / k) - np.floor((j - 1) * treatment_budget / k)
    priority_scores = R(env, s, a, y, infection_probs_predictor, transmission_probs_predictor, env.data_depth,
                        eta, beta)
    d = U(priority_scores, delta_j) + w
  return d


def update_alpha_and_zeta(alpha, zeta, j, tau=1, rho=1):
  """

  :param alpha:
  :param zeta:
  :param j:
  :param tau: Tuning parameter chosen with double bootstrap (?)
  :param rho: Tuning parameter chosen with double bootstrap (?)
  :return:
  """
  new_alpha = np.power(tau / (rho + j + 1), 1.25)
  new_zeta = 100.0 / (j + 1)
  return new_alpha, new_zeta


def tune_stochastic_approximation(s, gen_model_posterior, infection_probs_predictor):
  # See supplementary materials of RSS paper (ctrl + f ''tuning procedure'')
  B = 100
  T = 15  # Different T from time horizon!  Trying to be consistent with supplementary materials.
  RHO_GRID = np.linspace(0.1, 5, num=10)  # ToDo: How to choose these?
  TAU_GRID = np.linspace(0.1, 5, num=10)
  DELTA = [(rho, tau) for rho in RHO_GRID for tau in TAU_GRID]

  qhat_deltas = []
  for rho, tau in DELTA:
    qhat_delta = 0.0
    for b in range(B):
      beta_tilde = gen_model_posterior()
      # Initialize state and history
      s_b = s
      h_b = s_b
      for r in range(T):
        # ToDo: finish implementing
        pass

  best_qhat_ix = np.argmax(qhat_deltas)
  best_params = DELTA[best_qhat_ix]
  return best_params


def stochastic_approximation(T, s, y, beta, eta, f, g, alpha, zeta, tol, maxiter, dimension, treatment_budget,
                             k, feature_function, env, infection_probs_predictor, transmission_prob_predictor,
                             data_depth):
  """

  :param data_depth:
  :param transmission_prob_predictor:
  :param infection_probs_predictor:
  :param env:
  :param T:
  :param s:
  :param y:
  :param eta:
  :param f: Function to sample from conditional distribution of S.
  :param g: Function to sample from conditional distribution of Y.
  :param alpha:
  :param zeta:
  :param tol:
  :param maxiter:
  :param dimension: dimension of policy parameter
  :param treatment_budget:
  :param k: number of locations to change during decision rule iterations
  :param feature_function:
  :return:
  """
  DIFF_TOL = 0.0001

  it = 0
  a_dummy = np.zeros(env.L)  # ToDo: Figure out what this should be!  (input to feature function)
  diff = float('inf')
  while alpha > tol and it < maxiter and diff > DIFF_TOL:
    z = np.random.random(size=dimension)
    s_tpm = s
    y_tpm = y
    # x_tpm = feature_function(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor,
    #                          data_depth, beta)

    s_tpm_tilde = s
    y_tpm_tilde = y
    # x_tpm_tilde = feature_function(env, s_tpm_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
    #                                transmission_prob_predictor, data_depth, beta)

    s_tpmp1 = s_tpm
    s_tpmp1_tilde = s_tpm

    for m in range(T-1):
      # Plus perturbation
      eta_plus = eta + zeta * z
      priority_score_plus = R(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor,
                              data_depth, eta_plus, beta)
      a_tpm = decision_rule(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor, eta,
                            beta, k, treatment_budget, priority_score_plus)
      infection_probs = infection_probs_predictor(a_tpm, y_tpm, s_tpm, beta, 0.0, env.L, env.adjacency_list)
      y_tpm = np.random.binomial(n=1, p=infection_probs)
      # x_tpm = feature_function(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor,
      #                          transmission_prob_predictor, data_depth, beta)

      # Minus perturbation
      eta_minus = eta - zeta * z
      priority_score_minus = R(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor,
                               data_depth, eta_minus, beta)
      a_tpm_tilde = decision_rule(env, s_tpmp1_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
                                  transmission_prob_predictor, eta, beta, k, treatment_budget, priority_score_minus)
      infection_probs_tilde = infection_probs_predictor(a_tpm_tilde, y_tpm_tilde, s_tpm_tilde, beta, 0.0, env.L,
                                                        env.adjacency_list)
      y_tpm_tilde = np.random.binomial(n=1, p=infection_probs_tilde)
      # x_tpm_tilde = feature_function(env, s_tpmp1_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
      #                                transmission_prob_predictor, data_depth, beta)

      # Update states
      s_tpm = s_tpmp1
      s_tpm_tilde = s_tpmp1_tilde

    new_eta = update_eta(eta, alpha, zeta, z, y_tpm, y_tpm_tilde)
    diff = np.linalg.norm(eta - new_eta) / np.max((0.001, np.linalg.norm(eta)))
    eta = copy.copy(new_eta)
    alpha, zeta = update_alpha_and_zeta(alpha, zeta, it)
    it += 1
    # print('it: {}\nalpha: {}\nzeta: {}\neta: {}'.format(it, alpha, zeta, eta))

  return eta


"""
Implementing priority score below. See pdf pg. 15.
"""


def psi(infected_locations, predicted_infection_probs, lambda_, transmission_probabilities, data_depth):
  """
  Different from 'psi' for env-specific features!

  :param transmission_probabilities:
  :param infected_locations:
  :param predicted_infection_probs:
  :param lambda_: LxL matrix
  :param transmission_proba  :param m_hat: LxL matrix of estimated transmission probabilities under estimated modelbilities:

  :param data_depth: vector of [c^l] in paper
  :return:
  """
  psi_1 = predicted_infection_probs

  # Compute multiplier, not sure this is right
  transmission_probabilities_inf = transmission_probabilities[:, infected_locations[0]]
  lambda_inf = lambda_[:, infected_locations[0]]
  transmission_probs_times_lambda_inf = np.multiply(transmission_probabilities_inf, lambda_inf)
  multiplier = np.dot(transmission_probs_times_lambda_inf, 1 - predicted_infection_probs[infected_locations])

  psi_2 = np.multiply(psi_1, multiplier)
  psi_3 = np.multiply(psi_1, data_depth)

  return psi_1, psi_2, psi_3


def phi(infected_locations, lambda_, transmission_probabilities, psi_1, psi_2, data_depth):
  lambda_inf = lambda_[:, infected_locations]
  transmission_probabilities_inf = transmission_probabilities[:, infected_locations]

  psi_1_inf = psi_1[infected_locations]
  psi_2_inf = psi_2[infected_locations]
  data_depth_inf = data_depth[infected_locations]

  phi_1 = np.dot(lambda_inf, psi_1_inf)
  phi_2 = np.dot(transmission_probabilities_inf, psi_2_inf)
  phi_3 = np.dot(transmission_probabilities_inf, data_depth_inf)

  phi = np.column_stack((phi_1, phi_2, phi_3))

  return phi


def features_for_priority_score(env, s, a, y, infection_probs_predictor, transmission_prob_predictor, data_depth, beta):
  lambda_ = env.lambda_

  # Get predicted probabilities
  predicted_infection_probs = infection_probs_predictor(a, s, y, beta, 0.0, env.L, env.adjacency_list)
  transmission_probabilities = transmission_prob_predictor(a, beta, env.L, env.adjacency_matrix)

  # Get infection status-specific features
  infected_locations = np.where(y == 1)
  not_infected_locations = np.where(y == 0)
  psi_1, psi_2, psi_3 = psi(infected_locations, predicted_infection_probs, lambda_, transmission_probabilities,
                            data_depth)
  phi_ = phi(infected_locations, lambda_, transmission_probabilities, psi_1, psi_2, data_depth)

  # Collect features
  priority_score_features = np.zeros((env.L, 3))
  psi_inf = np.column_stack((psi_1, psi_2, psi_3))[infected_locations, :]
  phi_inf = phi_[not_infected_locations, :]
  priority_score_features[infected_locations, :] = psi_inf
  priority_score_features[not_infected_locations, :] = phi_inf

  return priority_score_features


def policy_search(env, time_horizon, gen_model_posterior,
                  initial_policy_parameter, initial_alpha, initial_zeta, infection_probs_predictor,
                  transmission_probs_predictor, treatment_budget, tol=1e-3, maxiter=100,
                  feature_function=features_for_priority_score, k=5):
  """
  Alg 1 on pg 10 of Nick's WNS paper; referring to parameter of transition model as 'beta', instead of 'eta'
  as in QL draft and the rest of this source code

  :param treatment_budget:
  :param infection_probs_predictor:
  :param transmission_probs_predictor:
  :param feature_function:
  :param maxiter:
  :param tol:
  :param initial_zeta:
  :param initial_alpha:
  :param env:
  :param time_horizon:
  :param gen_model_posterior: function that returns draws from conf dbn over gen model parameter
  :param initial_policy_parameter:
  :param k: number of locations to change during decision rule iterations
  :return:
  """

  dimension = len(initial_policy_parameter)
  beta_tilde = gen_model_posterior()

  policy_parameter = stochastic_approximation(time_horizon, env.current_state, env.current_infected, beta_tilde,
                                              initial_policy_parameter, infection_probs_predictor,
                                              transmission_probs_predictor, initial_alpha, initial_zeta, tol, maxiter,
                                              dimension, treatment_budget, k, feature_function, env,
                                              infection_probs_predictor, transmission_probs_predictor,
                                              env.data_depth)

  # Get priority function features
  a_for_transmission_probs = np.zeros(env.L)  # ToDo: Check which action is used to get transmission probs

  # ToDo: distinguish between env.pairwise_distances and Ebola.DISTANCE_MATRIX !
  # transmission_probabilities = transmission_probs_predictor(a_for_transmission_probs, beta_tilde, env.L,
  #                                                           env.adjacency_matrix)

  # infected_locations = np.where(env.current_infected == 1)
  # predicted_infection_probs = infection_probs_predictor(a_for_transmission_probs, env.current_infected,
  #                                                       env.current_state, beta_tilde, 0.0, env.L,
  #                                                       env.adjacency_list)
  features = feature_function(env, env.current_state, a_for_transmission_probs, env.current_infected,
                              infection_probs_predictor, transmission_probs_predictor, env.data_depth, beta_tilde)

  priority_scores = np.dot(features, policy_parameter)
  a_ix = np.argmax(priority_scores)
  a = np.zeros(env.L)
  a[a_ix] = 1
  return a


def policy_search_policy(**kwargs):
  # ToDo: Currently specific to SIS!

  env, T, treatment_budget = kwargs['env'], kwargs['planning_depth'], kwargs['treatment_budget']

  beta_mean = fit_infection_prob_model(env, None)
  beta_cov = env.mb_covariance(beta_mean)

  def gen_model_posterior():
    beta_tilde = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)
    return beta_tilde

  # Settings
  initial_policy_parameter = np.zeros(3)
  initial_alpha = initial_zeta = 1.0

  a = policy_search(env, T, gen_model_posterior,
                    initial_policy_parameter, initial_alpha, initial_zeta, sis_inf_probs.sis_infection_probability,
                    sis_inf_probs.get_all_sis_transmission_probs_omega0, treatment_budget, tol=1e-3, maxiter=100,
                    feature_function=features_for_priority_score, k=5)
  return a, None