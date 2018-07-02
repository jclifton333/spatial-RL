"""
Fit quadratic approximation to Q function fixed at observed states (for converting argmax Q to binary quadratic
program).
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import pdb


def get_neighbor_ixn_features(a, neighbor_interactions):
  """
  Return treatments at l's neighbors and all neighbor treatment interactions, i.e.
  [a_j a_j*a_k] for j,k in neighbors(a_l) including a_l
  """
  neighbor_ixn_features = [a[ixn[0]]*a[ixn[1]] for ixn in neighbor_interactions]
  return neighbor_ixn_features


def sample_from_q(q, treatment_budget, evaluation_budget, L):
  """
  Evaluate q function at evaluation_budget points in order to fit quadratic approximation.
  """
  sample_qs = []
  sample_acts = []
  dummy_act = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
  for sample in range(evaluation_budget):
    rand_act = np.random.permutation(dummy_act)
    q_sample = q(rand_act)
    sample_qs.append(q_sample)
    sample_acts.append(rand_act)
  return sample_qs, sample_acts


def fit_quad_approx_at_location(sample_qs, sample_acts, l, neighbor_interaction_lists):
  reg = LinearRegression()
  X = np.array([get_neighbor_ixn_features(a, neighbor_interaction_lists[l]) for a in sample_acts])
  y = sample_qs[:, l]
  reg.fit(X, y)
  return reg.intercept_, reg.coef_


def fit_quad_approx(sample_qs, sample_acts, neighbor_interaction_lists, L):
  quadratic_parameters = np.zeros((L,L))
  intercept = 0
  for l in range(L):
    intercept_l, beta_l = fit_quad_approx_at_location(sample_qs, sample_acts, l, neighbor_interaction_lists)
    neighbor_interactions = neighbor_interaction_lists[l]
    quadratic_parameters[neighbor_interactions[:,0], neighbor_interactions[:,1]] += beta_l
    intercept += intercept_l
  return quadratic_parameters, intercept


def get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env):
  sample_qs, sample_acts = sample_from_q(q, treatment_budget, evaluation_budget, env.L)
  quadratic_parameters, intercept = fit_quad_approx(sample_qs, sample_acts, env.neighbor_interaction_lists, env.L)
  return quadratic_parameters, intercept






