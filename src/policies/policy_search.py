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
import numpy as np


def R(env, s, a, y, infection_probs_predictor, transmission_prob_predictor, data_depth, eta):
  """
  Linear priority score function.

  """
  priority_features = features_for_priority_score(env, s, a, y, infection_probs_predictor, transmission_prob_predictor,
                                                  data_depth)
  return np.dot(priority_features, eta)


def update_eta(eta, alpha, zeta, z, y, y_tilde):
  ones = np.ones(len(y))
  second_term = z * np.dot(ones, y - y_tilde)
  new_eta = eta + alpha / (2 * zeta) * second_term
  new_eta_norm = np.linalg.norm(new_eta)
  new_eta /= new_eta_norm  # Project onto unit sphere.
  return new_eta


def U(priority_scores, m):
  """

  :param priority_scores: Corresponds to R above.
  :param m: Integer >= 1.
  :return:
  """
  priority_scores_mth_order_stat = np.argsort(priority_scores)[m]  # ToDo: Optimize?
  U = priority_scores >= priority_scores_mth_order_stat
  return U


def decision_rule(s, priority_scores, treatment_budget, k):
  d = np.zeros(len(priority_scores))
  floor_c_by_k = np.floor(treatment_budget / k)
  d[np.argsort(-priority_scores)[:floor_c_by_k]] = 1
  for j in range(1, k):
    w = d
    delta_j = np.floor(j * treatment_budget / k) - np.floor((j - 1) * treatment_budget / k)
    priority_scores = R(s, w, eta)
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
  new_alpha = np.power(tau / (rho + j), 1.25)
  new_zeta = 100.0 / j
  return new_alpha, new_zeta


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
  it = 0
  a_dummy = np.zeros(env.L)  # ToDo: Figure out what this should be!  (input to feature function)
  while alpha > tol and it < maxiter:
    z = np.random.random(size=dimension)
    s_tpm = s
    y_tpm = y
    # x_tmp = feature_function(env, s_tpm, a_dummy, y_tpm, infection_probs_predictor, transmission_prob_predictor,
    #                          data_depth)

    s_tpm_tilde = s
    y_tpm_tilde = y
    x_tpm_tilde = feature_function(env, s_tpm_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
                                   transmission_prob_predictor, data_depth, beta)

    for m in range(T-1):
      eta_plus = eta + zeta * z
      priority_score_plus = R(x_tpm, eta_plus)
      a_tpm = decision_rule(x_tpm, priority_score_plus, treatment_budget, k)
      s_tpmp1 = g(s_tpm, a_tpm)
      y_tpm = f(s_tpm, a_tpm)
      x_tpm = feature_function(env, s_tpmp1, a_dummy, y_tpm, infection_probs_predictor,
                               transmission_prob_predictor, data_depth, beta)
      eta_minus = eta - zeta * z
      priority_score_minus = R(x_tpm_tilde, eta_minus)
      a_tpm_tilde = decision_rule(s_tpm_tilde, priority_score_minus, treatment_budget, k)
      y_tpm_tilde = f(s_tpm_tilde, a_tpm_tilde)
      s_tpmp1_tilde = g(s_tpm_tilde, a_tpm_tilde)
      x_tpm_tilde = feature_function(env, s_tpmp1_tilde, a_dummy, y_tpm_tilde, infection_probs_predictor,
                                     transmission_prob_predictor, data_depth, beta)
    eta = update_eta(eta, alpha, zeta, z, y_tpm, y_tpm_tilde)
    alpha, zeta = update_alpha_and_zeta(alpha, zeta, j)
    it += 1

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
  psi_1_inf = psi_1[infected_locations]
  transmission_probabilities_inf = transmission_probabilities[:, infected_locations]
  lambda_inf = lambda_[:, infected_locations]
  transmission_probs_times_lambda_inf = np.multiply(transmission_probabilities_inf, lambda_inf)
  multiplier = np.dot(transmission_probs_times_lambda_inf, 1 - psi_1_inf)

  psi_2 = np.multiply(psi_1_inf, multiplier)
  psi_3 = np.multiply(psi_1, data_depth)

  return psi_1, psi_2, psi_3


def phi(not_infected_locations, lambda_, transmission_probabilities, psi_1, psi_2, data_depth):
  lambda_not_inf = lambda_[:, not_infected_locations]
  transmission_probabilities_not_inf = transmission_probabilities[:, not_infected_locations]

  psi_1_not_inf = psi_1[not_infected_locations]
  psi_2_not_inf = psi_2[not_infected_locations]
  data_depth_not_inf = data_depth[not_infected_locations]

  phi_1 = np.dot(lambda_not_inf, psi_1_not_inf)
  phi_2 = np.dot(transmission_probabilities_not_inf, psi_2_not_inf)
  phi_3 = np.dot(transmission_probabilities_not_inf, data_depth_not_inf)

  return np.array([phi_1, phi_2, phi_3])


def features_for_priority_score(env, s, a, y, infection_probs_predictor, transmission_prob_predictor, data_depth, beta):
  lambda_ = env.lambda_

  # Get predicted probabilities
  predicted_infection_probs = infection_probs_predictor(a, s, y, beta, 0.0, env.L, env.adjacency_list)
  transmission_probabilities = transmission_prob_predictor(s, a, y)

  # Get infection status-specific features
  infected_locations = np.where(y == 1)
  not_infected_locations = np.where(y == 0)
  psi_1, psi_2, psi_3 = psi(infected_locations, predicted_infection_probs, lambda_, transmission_probabilities,
                            data_depth)
  phi_ = phi(not_infected_locations, lambda_, transmission_probabilities, psi_1, psi_2, data_depth)

  # Collect features
  priority_score_features = np.zeros((env.L, 3))
  psi_inf = np.column_stack((psi_1, psi_2, psi_3))[infected_locations, :]
  phi_inf = phi_[not_infected_locations, :]
  priority_score_features[infected_locations, :] = psi_inf
  priority_score_features[not_infected_locations, :] = phi_inf

  return priority_score_features


def policy_search(env, time_horizon, beta_post_mean, beta_post_var,
                  initial_policy_parameter, initial_alpha, initial_zeta, infection_probs_predictor,
                  transmission_probs_predictor, treatment_budget, data_depth=1, tol=1e-3, maxiter=100,
                  feature_function=features_for_priority_score, k=5):
  """
  Alg 1 on pg 10 of Nick's WNS paper; referring to parameter of transition model as 'beta', instead of 'eta'
  as in QL draft and the rest of this source code

  :param data_depth:
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
  :param beta_post_mean: mean of (normal) confidence dbn for gen model parameters
  :param beta_post_var: var of (normal) confidence dbn for gen model parameters
  :param initial_policy_parameter:
  :param k: number of locations to change during decision rule iterations
  :return:
  """

  dimension = len(initial_policy_parameter)
  beta_tilde = np.random.multivariate_normal(beta_post_mean, beta_post_var)

  policy_parameter = stochastic_approximation(time_horizon, env.current_state, env.current_infected,
                                              initial_policy_parameter, infection_probs_predictor,
                                              transmission_probs_predictor, initial_alpha, initial_zeta, tol, maxiter,
                                              dimension, treatment_budget, k, feature_function, env,
                                              infection_probs_predictor, transmission_probs_predictor,
                                              data_depth, beta_tilde)

  # Get priority function features
  a_for_transmission_probs = np.zeros(env.L)  # ToDo: Check which action is used to get transmission probs

  # ToDo: distinguish between env.pairwise_distances and Ebola.DISTANCE_MATRIX !
  transmission_probabilities = transmission_probs_predictor(a_for_transmission_probs, beta_tilde, env.L,
                                                            env.adjacency_matrix)
  for l in range(env.L):
    for lprime in range(env.L):
      transmission_probabilities[l, lprime] = transmission_probs_predictor(a_for_transmission_probs, l, lprime,
                                                                           beta_tilde, env.DISTANCE_MATRIX)

  infected_locations = np.where(env.current_infections == 1)
  predicted_infection_probs = infection_probs_predictor(env, beta_tilde)
  transmission_probabilities = transmission_probs_predictor(env, beta_tilde)
  features = psi(infected_locations, predicted_infection_probs, env.lambda_, transmission_probabilities, env.data_depth)

  priority_scores = np.dot(features, policy_parameter)
  a = np.argmax(priority_scores)

  return a


