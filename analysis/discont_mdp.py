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
from scipy.stats import multivariate_normal
from scipy.integrate import nquad


def conditional_expectation_linear_inequality(a, b, mean_vector, sigma_sq):
  """
  Compute expectation of dot(a, Y) given dot(a - b, Y) >= 0, where
  Y ~ N(mean_vector, sigma_sq * eye(4).
  :param a:
  :param b:
  :param sigma_sq:
  :return:
  """
  amb = a - b
  normal_pdf = lambda x, mu, s: (1 / np.sqrt(2*np.pi*s)) * np.exp(-0.5 * (x - mu)**2 / s)
  joint_pdf = lambda x1, x2, x3, x4: normal_pdf(x1, mean_vector[0], sigma_sq) * \
    normal_pdf(x2, mean_vector[1], sigma_sq) * normal_pdf(x3, mean_vector[2], sigma_sq) * \
    normal_pdf(x4, mean_vector[3], sigma_sq)
  lim0 = lambda x1, x2, x3: [-(amb[0]*x1 + amb[1]*x2 + amb[2]*x3), float('inf')]
  limits = [lim0, [-float('inf'), float('inf')], [-float('inf'), float('inf')], [-float('inf'), float('inf')]]
  solution = nquad(joint_pdf, limits)
  return solution[0]


def mc_conditional_expectation_linear_inequality(a, b, mean_vector, sigma_sq, mc_replicates=10000):
  draws = np.random.multivariate_normal(mean_vector, sigma_sq*np.eye(4), size=mc_replicates)
  conditioning_indicator = np.dot(draws, a-b) >= 0
  conditional_draws = draws[np.where(conditioning_indicator) == 1, :]
  return np.mean(np.dot(conditional_draws, a))


def draw_q1_estimates(theta, eta, sigma_sq, X1, A1, mc_replicates=1000):
  """
  Draw many estimates of q1 by drawing from conditional distribution of next state at stage-1 obs (X1, A1).

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
    x1_a1 = np.column_stack((x1, np.multiply(x1, np.array([1, 0]))))
    x1_a2 = np.column_stack((x1, np.multiply(x1, np.array([0, 1]))))
    x21_a1_hats = np.dot(x1_a1, estimated_transition_parameters_1)
    x21_a2_hats = np.dot(x1_a2, estimated_transition_parameters_2)
    indicator = x21_a1_hats.sum(axis=0) >= x21_a2_hats.sum(axis=0)
    pseudo_outcome_i = np.multiply(x21_a1_hats, indicator) + np.multiply(x21_a2_hats, 1 - indicator)
    pseudo_outcomes.append(pseudo_outcome_i)

  # Fit pseudo-outcome models for each draw
  # ToDo: Note that this linear model is misspecified!
  pseudo_outcomes_stack = np.vstack(pseudo_outcomes)
  estimated_q1_parameters = np.dot(Xprime_Xinv_Xprime, pseudo_outcomes_stack)

  return estimated_q1_parameters


def get_stage_one_actions_under_q1_estimates(x, mu_1, mu_2, estimated_q1_parameters):
  """
  Get actions at estimated q1 parameter at stage-1 state x.

  :param x:
  :param mu_1:
  :param mu_2:
  :param estimated_q1_parameters:
  :return:
  """
  x_a1 = np.concatenate(x, np.multiply(x, np.array([1, 0])))
  x_a2 = np.concatenate(x, np.multiply(x, np.array([0, 1])))

  # Get estimates of next-stage rewards under optimal policy
  predicted_stage1_outcomes_a1 = np.dot(x_a1, estimated_q1_parameters)
  indicator_a1 = predicted_stage1_outcomes_a1[0, :] >= predicted_stage1_outcomes_a1[1, :]
  expected_reward_a1 = np.multiply(predicted_stage1_outcomes_a1[0, :], indicator_a1) + \
                       np.multiply(predicted_stage1_outcomes_a1[1, :], 1 - indicator_a1)
  predicted_stage1_outcomes_a2 = np.dot(x_a2, estimated_q1_parameters)
  indicator_a2 = predicted_stage1_outcomes_a2[0, :] >= predicted_stage1_outcomes_a2[1, :]
  expected_reward_a2 = np.multiply(predicted_stage1_outcomes_a2[0, :], indicator_a2) + \
                       np.multiply(predicted_stage1_outcomes_a2[1, :], 1 - indicator_a2)

  # Get estimated q1's
  q_a1_hats = mu_1 + expected_reward_a1
  q_a2_hats = mu_2 + expected_reward_a2

  # Get actions taken at each replicates
  actions = q_a1_hats >= q_a2_hats

  return actions


if __name__ == "__main__":
  mc_conditional_expectation_linear_inequality(np.random.random(size=4), np.random.random(size=4),
                                               np.random.random(size=4), 1.0)










