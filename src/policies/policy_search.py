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

ToDo: || eta || = 1, so need to project
"""
import numpy as np


def features_for_priority_score(x):
  # ToDo: PLACEHOLDER
  return x


def R(x, eta):
  """
  Linear priority score function.

  :param x:
  :param eta:
  :return:
  """
  priority_features = features_for_priority_score(x)
  return np.dot(priority_features, eta)


def update_eta(eta, alpha, zeta, z, y, y_tilde):
  ones = np.ones(len(y))
  second_term = z * np.dot(ones, y - y_tilde)
  new_eta = eta + alpha / (2 * zeta) * second_term
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


def update_alpha_and_zeta(alpha, zeta):
  return alpha, zeta


def stochastic_approximation(T, s, a, y, eta, f, g, alpha, zeta, tol, maxiter, dimension, treatment_budget,
                             k, feature_function):
  """

  :param T:
  :param s:
  :param a:
  :param y:
  :param eta:
  :param f: Function to sample from conditional distribution of S.
  :param g: Function to sample from conditional distribution of Y.
  :param alpha:
  :param zeta:
  :param tol:
  :param maxiter:
  :param dimension:
  :param treatment_budget:
  :param k: number of locations to change during decision rule iterations
  :param feature_function:
  :return:
  """
  it = 0
  while alpha > tol and it < maxiter:
    z = np.random.random(size=dimension)
    s_tpm = s
    y_tpm = y
    x_tpm = feature_function(s_tpm, y_tpm)

    s_tpm_tilde = s
    y_tpm_tilde = y
    x_tpm_tilde = feature_function(s_tpm_tilde, y_tpm_tilde)
    for m in range(T-1):
      eta_plus = eta + zeta * z
      priority_score_plus = R(x_tpm, eta_plus)
      a_tpm = decision_rule(x_tpm, priority_score_plus, treatment_budget, k)
      s_tpmp1 = g(s_tpm, a_tpm)
      y_tpm = f(s_tpm, a_tpm)
      x_tpm = feature_function(s_tpmp1, y_tpm)

      eta_minus = eta - zeta * z
      priority_score_minus = R(x_tpm_tilde, eta_minus)
      a_tpm_tilde = decision_rule(s_tpm_tilde, priority_score_minus, treatment_budget, k)
      y_tpm_tilde = f(s_tpm_tilde, a_tpm_tilde)
      s_tpmp1_tilde = g(s_tpm_tilde, a_tpm_tilde)
      x_tpm_tilde = feature_function(s_tpmp1_tilde, y_tpm_tilde)

    eta = update_eta(eta, alpha, zeta, z, y_tpm, y_tpm_tilde)
    alpha, zeta = update_alpha_and_zeta(alpha, zeta)
    it += 1

  return eta


"""
Implementing priority score below. See pdf pg. 15.
"""


def psi(s, a, y, predict_infection_probs, lambda_, m_hat, data_depth):
  """
  Different from 'psi' for env-specific features!

  :param s:
  :param a:
  :param y:
  :param predict_infection_probs:
  :param lambda_: LxL matrix
  :param m_hat: corresponds to m_l,j in paper
  :param data_depth: vector of [c^l] in paper
  :return:
  """
  infected_locations = np.where(y == 1)
  psi_1 = predict_infection_probs(s, a, y)

  # Compute multiplier, not sure this is right
  psi_1_inf = psi_1[infected_locations]
  m_inf = m_hat(s, a)[infected_locations]
  lambda_inf = lambda_[infected_locations]
  multiplier = np.dot(lambda_, np.multiply(1 - psi_1_inf, m_inf, lambda_inf))

  psi_2 = np.multiply(psi_1, multiplier)
  psi_3 = np.multiply(psi_1, data_depth)

  return psi_1, psi_2, psi_3


def phi(s, a, y, lambda_, m_hat, psi_1, psi_2, data_depth):
  not_infected_locations = np.where(y == 0)
  lambda_not_inf = lambda_[not_infected_locations, not_infected_locations]
  psi_1_not_inf = psi_1[not_infected_locations]
  psi_2_not_inf = psi_2[not_infected_locations]

  phi_1 = np.dot(lambda_not_inf, psi_1_not_inf)
  phi_2 = None
  phi_3 = None

  return np.array([phi_1, phi_2, phi_3])


