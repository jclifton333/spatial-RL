"""
Fit quadratic approximation to Q function fixed at observed states (for approximating argmax Q as binary quadratic
program).
"""
import time
import pdb
import numpy as np
from src.estimation.optim.sweep.argmaxer_sweep import perturb_action
from sklearn.linear_model import LinearRegression
from numba import njit, jit


# @njit
# @jit
def get_neighbor_ixn_features(a, neighbor_interactions):
  """
  Return treatments at l's neighbors and all neighbor treatment interactions, i.e.
  [a_j a_j*a_k] for j,k in neighbors(a_l) including a_l
  """
  neighbor_ixn_features = []
  for i in range(len(neighbor_interactions)):
    ixn = neighbor_interactions[i]
    neighbor_ixn_features.append(a[ixn[0]]*a[ixn[1]])
  return neighbor_ixn_features


def sample_from_q(q, treatment_budget, evaluation_budget, L, initial_act):
  """
  Evaluate q function at evaluation_budget points in order to fit quadratic approximation.
  """
  if initial_act is not None:
    num_to_perturb = int(np.ceil(treatment_budget / 2))
    acts_to_evaluate = [perturb_action(initial_act, num_to_perturb) for e in range(evaluation_budget - 1)]
    acts_to_evaluate.append(initial_act)
  else:
    dummy_act = np.hstack((np.ones(treatment_budget), np.zeros(L - treatment_budget)))
    acts_to_evaluate = [np.random.permutation(dummy_act) for e in range(evaluation_budget)]
  sample_qs = []
  for ix, act in enumerate(acts_to_evaluate):
    sample_qs.append(q(act))
    print('evluating q function on sample {}'.format(ix))
  return sample_qs, acts_to_evaluate


def fit_quad_approx_at_location(sample_qs, sample_acts, l, l_ix, neighbor_interactions):
  reg = LinearRegression()
  X = np.array([get_neighbor_ixn_features(a, neighbor_interactions) for a in sample_acts])
  y = sample_qs[:, l_ix]
  reg.fit(X, y)
  return reg.intercept_, reg.coef_


def fit_quad_approx(sample_qs, sample_acts, neighbor_interaction_lists, env_L, ixs):
  quadratic_parameters = np.zeros((env_L, env_L))
  intercept = 0
  if ixs is None:
    ixs = range(env_L)
  for l_ix in range(len(ixs)):
    l = ixs[l_ix]
    neighbor_interactions = neighbor_interaction_lists[l]
    if len(neighbor_interactions) > 0:
      intercept_l, beta_l = fit_quad_approx_at_location(sample_qs, sample_acts, l, l_ix, neighbor_interactions)
      quadratic_parameters[neighbor_interactions[:, 0], neighbor_interactions[:, 1]] += beta_l
      intercept += intercept_l
  return quadratic_parameters, intercept


def get_quadratic_program_from_q(q, treatment_budget, evaluation_budget, env, ixs, initial_act=None):
  if ixs is not None:
    L = len(ixs)
  else:
    L = env.L
  sample_qs, sample_acts = sample_from_q(q, treatment_budget, evaluation_budget, L, initial_act)
  quadratic_parameters, intercept = fit_quad_approx(sample_qs, sample_acts, env.neighbor_interaction_lists, env.L, ixs)
  return quadratic_parameters, intercept






