"""
To see the effects of model misspecification on MB vs. MF approaches, we examine models of the form
\epsilon * SIS_transition_probs + (1 - \epsilon) * contamination_transition_probs.

contamination_transition_probs will be output by a neural network, which is tuned as follows:

1. Estimate r(0) = MSE( \hat{p}_MB | \epsilon=0) / MSE( \hat{p}_MF | \epsilon=0 ), i.e. the relative performance of
   MB and MF one-step probability estimates when the sis model is uncontaminated, for a size 50 lattice integrated
   over 25 time steps.
2. For \epsilon = 0.25, 0.75, 1, randomly generate contamination network parameters, and estimate relative MSEs
   for each.
3. Use ?
   to find contamination parameter that minimizes \lVert r(\epsilon; \beta) - r(\epsilon) \rVert^2,
   where r(\epsilon; \beta) is the observed ratio of MSEs and r(\epsilon) is the desired ratio - a line
   with slope (2 - r(0)) so that at full maximum contamination, MF MSE is half that of MB.
"""
import pdb
# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)
import numpy as np
from src.environments import generate_network, sis
from src.environments.sis_contaminator import SIS_Contaminator
from src.estimation.optim.argmaxer_factory import argmaxer_random
from src.estimation.q_functions.q_functions import q_max_all_states
from src.estimation.optim.quad_approx.argmaxer_quad_approx import argmaxer_quad_approx
from scipy.special import logit

SEED = 3


def tune():
  env = sis.SIS(L = 100, omega = 0.0, generate_network = generate_network.lattice)

  # Generate some data
  np.random.seed(SEED)
  dummy_action = np.append(np.ones(5), np.zeros(95))
  env.reset()
  for t in range(10):
    env.step(np.random.permutation(dummy_action))

  # Tune parameter
  contaminator = SIS_Contaminator()
  expit_parameter_0 = np.array([0.3, 0.4, 0.1, 0.2, 0.65, 0.7, 0.5, 0.6])
  parameter_0 = logit(expit_parameter_0 / 4)
  parameter_1 = logit(expit_parameter_0)
  parameter = np.concatenate((parameter_0, parameter_1, [0.0]))
  best_diff = 0.0
  best_param = parameter
  for i in range(20):
    q = contaminator.predict_proba

    # Evaluate random policy
    random_score_list = []
    for rep in range(10):
      score_list, _, _ = q_max_all_states(env, 100, 5, q, argmaxer_random)
      random_score_list.append(np.mean(score_list))
    random_score = np.mean(random_score_list)

    # Evaluate best policy
    best_score_list, _, _ = q_max_all_states(env, 100, 5, q, argmaxer_quad_approx)
    best_score = np.mean(best_score_list)
    print('random score: {} best score: {}'.format(random_score, best_score))

    if best_score - random_score < best_diff:
      best_diff = best_score - random_score
      best_param = parameter

    # Perturb
    parameter[:-1] = best_param[:-1] + np.random.normal(loc=0.0, scale=1, size=16)
    contaminator.set_weights(parameter)
  return


if __name__ == "__main__":
  tune()
