"""
# Setup: two-stage MDP where Q-function is additive over 2 locations, and the treatment budget is 1 location.
# Stage-1 actions (1, 0), (0, 1) have expected rewards mu.1, mu.2 resp.  Call these actions a1, a2.
# Expected payoff at stage 2 at each location is a linear function (with parameter \theta) of a
# stage-1 covariate and the stage-1 action.
# We want to evaluate the regret of the policy based on the estimator \theta, and in particular how this regret
# varies as \theta approaches boundaries separating the value of different actions.

# MDP
# States at stage 1 are randomly generated iid normal.  Stage 1 expected rewards are known constants mu.1, mu.2
# corresponding
# to the respective actions and independent of state.
# Stages at stage 2 conditional on stage 1 action and states are independent normal with mean vector [x1.i a1.i*x1.i]_
# {i=1,2} %*% (theta, eta)
# and variance sigma.sq.  Finally, stage 2 rewards are the component of the mean vector corresponding to the location
# that was chosen to
# be treated at stage 2.
"""
import numpy as np


def value_of_estimated_policy(theta, eta, mu_1, mu_2, sigma_sq, x, X1, A1, mc_replicates=1000):
  """
  Get expected value of policy estimated from first-stage observations (X1, A1).

  Local reward currently given by first component of state vector.

  :param theta: rows [theta1, theta2] controlling conditional distribution of features 1, 2 resp
  :param eta: rows [eta1, eta2] controlling conditional distribution of features 1, 2 resp
  :param mu_1:
  :param mu_2:
  :param sigma_sq:
  :param x:
  :param X1: list of arrays of stage-1 states
  :param A1: list of arrays of stage-1 actions
  :return:
  """
  X1_stack = np.vstack(X1)
  A1_stack = np.hstack(A1)
  design_matrix = np.column_stack(X1_stack, np.multiply(X1_stack, A1_stack))
  transition_parameter = np.column_stack((theta, eta))

  # Draw many reps from conditional dbn of next state
  conditional_means_1 = np.dot(design_matrix, transition_parameter[0, :])
  conditional_means_2 = np.dot(design_matrix, transition_parameter[1, :])
  X2 = np.array([
    np.column_stack((np.random.normal(loc=conditional_means_1, scale=np.sqrt(sigma_sq)),
                     np.random.normal(loc=conditional_means_2, scale=np.sqrt(sigma_sq))))
    for _ in range(mc_replicates)]).T

  # Estimate transition_parameter on each draw
  Xprime_Xinv_Xprime = np.dot(np.linalg.inv(np.dot(design_matrix.T, design_matrix)), design_matrix.T)
  estimated_transition_parameters_1 = np.dot(Xprime_Xinv_Xprime, X2[:, 0])
  estimated_transition_parameters_2 = np.dot(Xprime_Xinv_Xprime, X2[:, 1])

  # Get pseudo-outcome for each draw
  pseudo_outcomes = []
  for x1 in X1:
    x1_a1 = np.column_stack((x, np.multiply(x1, np.array([1, 0]))))
    x1_a2 = np.column_stack((x1, np.multiply(x1, np.array([0, 1]))))
    x21_a1_hats = np.dot(x1_a1, estimated_transition_parameters_1)
    x21_a2_hats = np.dot(x1_a2, estimated_transition_parameters_1)
    indicator = x21_a1_hats.sum(axis=0) > x21_a2_hats.sum(axis=0)
    pseudo_outcome_i = np.multiply(x21_a1_hats, indicator) + np.multiply(x21_a2_hats, 1 - indicator)
    pseudo_outcomes.append(pseudo_outcome_i)

  # Fit pseudo-outcome models for each draw












