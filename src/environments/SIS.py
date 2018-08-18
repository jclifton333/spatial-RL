"""
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
"""

import copy
import numpy as np
from src.policies.helpers import expit_derivative
from .SpatialDisease import SpatialDisease
from .sis_contaminator import SIS_Contaminator
from .sis_infection_probs import sis_infection_probability
from scipy.special import expit
from scipy.linalg import block_diag
import pdb
import networkx as nx

import os
this_dir = os.path.dirname(os.path.abspath(__file__))
tuning_data_dir = os.path.join(this_dir, 'tuning', 'tuning_data')


class SIS(SpatialDisease):
  # PATH_LENGTH = 2 # For path-based features
  # POWERS_OF_TWO_MATRICES ={
  #   k: np.array([[np.power(2.0, 3*i-j) for j in range(1, 3+1)] for i in range(1, k + 1)]) for k in range(1, PATH_LENGTH + 1)
  # }
  ENCODING_MATRIX = np.array([np.power(2.0, 3-j) for j in range(1, 3+1)])


  # Fixed generative model parameters
  BETA_0 = 0.9
  BETA_1 = 1.0
  BETA = np.array([BETA_0, BETA_1])
  INITIAL_INFECT_PROB = 0.1

  """
  Parameters for infection probabilities.  See
    bin/run_infShieldState.cpp
    main/infShieldStatePosImNoSoModel.cpp
  
  Correspondence between draft and stmMF_cpp parameter names
    intcp_inf_latent_ : ETA_0
    trt_pre_inf_      : ETA_1, ETA_3
    intcp_inf_        : ETA_2
    trt_act_inf_      : ETA_4
    intcp_rec_        : ETA_5
    trt_act_rec_      : ETA_6
  """
  PROB_INF_LATENT = 0.01
  PROB_INF = 0.5
  PROB_NUM_NEIGH = 3
  PROB_REC = 0.25

  ETA_0 = np.log(1 / (1 - PROB_INF_LATENT) - 1)
  ETA_2 = np.log(((1 - PROB_INF) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1)
  ETA_4 = np.log(((1 - PROB_INF * 0.25) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1) - ETA_2
  ETA_3 = np.log(((1 - PROB_INF * 0.75) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1) - ETA_2
  ETA_5 = np.log(1 / (1 - PROB_REC) - 1)
  ETA_6 = np.log(1 / ((1 - PROB_REC) * 0.5) - 1) - ETA_5
  ETA = np.array([ETA_0, ETA_3, ETA_2, ETA_3, ETA_4, ETA_5, ETA_6])

  # Contamination model stuff
  # CONTAMINATION_MODEL_FNAME = os.path.join(tuning_data_dir, 'initial-sample-ratios-and-losses-180729_172420.p')
  # CONTAMINATION_MODEL_DATA = pkl.load(open(CONTAMINATION_MODEL_FNAME, 'rb'))
  # CONTAMINATION_MODEL_PARAMETER = CONTAMINATION_MODEL_DATA[3]['contamination_model_parameter']
  # CONTAMINATION_MODEL_PARAMETER = [CONTAMINATION_MODEL_PARAMETER[:-1].reshape(-1, 1),
  #                                  CONTAMINATION_MODEL_PARAMETER[-1].reshape(1)]
  CONTAMINATOR = SIS_Contaminator()
  CONTAMINATION_MODEL_PARAMETER = np.array([
    0.99, 0.32, 0.15, -0.83, -0.03, -0.07, 0.06, -0.21, 0.08, -0.14, -0.56, 0.54, 0.54, 0.95, 0.13, -0.10, -2.2
  ])
  CONTAMINATOR.set_weights(CONTAMINATION_MODEL_PARAMETER, 16)

  def __init__(self, L, omega, generate_network, add_neighbor_sums=False, adjacency_matrix=None,
               initial_infections=None, initial_state=None, eta=None, beta=None,
               epsilon=0, contaminator=CONTAMINATOR):
    """
    :param omega: parameter in [0,1] for mixing two SIS models
    :param generate_network: function that accepts network size L and returns adjacency matrix
    """
    self.ENCODING_DICT = {
          s: {
            a: {
              y:
                int(np.dot(np.array([s, a, y]), SIS.ENCODING_MATRIX)) for y in range(2)
            }
            for a in range(2)
          }
          for s in range(2)
    }
    self.add_neighbor_sums = add_neighbor_sums
    self.epsilon = epsilon
    self.contaminator = contaminator

    if eta is None:
      self.eta = SIS.ETA
    else:
      self.eta = eta

    if beta is None:
      self.beta = SIS.BETA
    else:
      self.beta = beta

    if adjacency_matrix is None:
      self.adjacency_matrix = generate_network(L)
    else:
      self.adjacency_matrix = adjacency_matrix
    SpatialDisease.__init__(self, self.adjacency_matrix, initial_infections)

    if initial_state is None:
      self.initial_state = np.zeros(self.L)
    else:
      self.initial_state = initial_state

    self.omega = omega
    self.state_covariance = self.beta[1] * np.eye(self.L)

    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = [] # Network-level features
    self.current_state = self.S[-1, :]

    # These are for efficiently computing gradients for estimating generative model
    self.max_num_neighbors = int(np.max(np.sum(self.adjacency_matrix, axis=0)))
    self.counts_for_likelihood_next_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    self.counts_for_likelihood_next_not_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))

    # These are for figuring out which bootstrap weights correspond to the not-infected locations
    self.indices_for_likelihood_next_infected = {i: {j:{k:[] for k in range(self.max_num_neighbors + 1)}
                                                     for j in range(self.max_num_neighbors + 1)}
                                                 for i in range(2)}
    self.indices_for_likelihood_next_not_infected = {i: {j:{k:[] for k in range(self.max_num_neighbors + 1)}
                                                     for j in range(self.max_num_neighbors + 1)}
                                                 for i in range(2)}

  def reset(self):
    """
    Reset state and observation histories.
    """
    super(SIS, self).reset()
    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = []
    self.current_state = self.S[-1,:]
    self.counts_for_likelihood_next_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    self.counts_for_likelihood_next_not_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    self.indices_for_likelihood_next_infected = {i: {j:{k:[] for k in range(self.max_num_neighbors + 1)}
                                                     for j in range(self.max_num_neighbors + 1)}
                                                 for i in range(2)}
    self.indices_for_likelihood_next_not_infected = {i: {j:{k:[] for k in range(self.max_num_neighbors + 1)}
                                                     for j in range(self.max_num_neighbors + 1)}
                                                 for i in range(2)}

  ##############################################################
  ##            Feature function computation                  ##
  ##############################################################

  def psi_at_location(self, l, data_block):
    # psi_l = np.zeros(16)

    # # Get encoding for location l
    # row_l = data_block[l, :]
    # ix = int(np.dot(row_l, SIS.ENCODING_MATRIX))
    # psi_l[ix] = 1

    # # Get encodings for l's neighbors
    # for lprime in self.adjacency_list[l]:
    #   row_lprime = data_block[lprime, :]
    #   ix = int(np.dot(row_lprime, SIS.ENCODING_MATRIX))
    #   psi_l[8 + ix] += 1
    s, a, y = data_block[l, :]
    psi_l = [0]*8
    psi_l[self.ENCODING_DICT[int(s)][int(a)][int(y)]] = 1
    psi_neighbors = [0]*8
    for lprime in self.adjacency_list[l]:
      s, a, y = data_block[lprime, :]
      psi_neighbors[self.ENCODING_DICT[int(s)][int(a)][int(y)]] += 1
    return np.concatenate((psi_l, psi_neighbors))

  def psi(self, data_block):
    """
    :param data_block:
    :return:
    """
    data_block[:, -1] = 1 - data_block[:, -1]
    psi = np.zeros((0, 16))
    for l in range(self.L):
      psi_l = self.psi_at_location(l, data_block)
      psi = np.vstack((psi, psi_l))
    return psi

  def psi_at_action(self, old_raw_data_block, old_data_block, old_action, action):
    new_data_block = copy.copy(old_data_block)
    if self.add_neighbor_sums:
      new_data_block = new_data_block[:, :int(new_data_block.shape[1] / 2)]
    locations_with_changed_actions = set(np.where(old_action != action)[0])

    for l in range(self.L):
      l_and_neighbors = [l] + self.adjacency_list[l]
      if self.is_any_element_in_set(l_and_neighbors, locations_with_changed_actions):
        new_data_block[l, :] = self.psi_at_location(l, old_raw_data_block)
    if self.add_neighbor_sums:
      new_data_block = self.psi_neighbor(new_data_block)
    return new_data_block

  def psi_neighbor(self, data_block):
    """
    To psi features, concatenate another block of features which are the _sums of the neighboring psi features_.
    :param data_block:
    :return:
    """
    neighbor_data_block = np.zeros(data_block.shape)
    for l in range(self.L):
      neighbor_data_block[l] = np.sum(data_block[self.adjacency_list[l],:], axis=0)
    return np.column_stack((data_block, neighbor_data_block))

  ##############################################################
  ##            End path-based feature function stuff         ##
  ##############################################################

  def add_state(self, s):
    self.S = np.vstack((self.S, s))
    self.S_indicator = np.vstack((self.S_indicator, s > 0))
    self.current_state = s

  def next_state(self):
    """
    Update state array acc to AR(1)
    :return next_state: self.L-length array of new states
    """
    super(SIS, self).next_state()
    next_state = np.random.multivariate_normal(mean=self.beta[0]*self.current_state, cov=self.state_covariance)
    self.add_state(next_state)
    return next_state

  def infection_probability(self, a, y, s, eta=ETA):
    return sis_infection_probability(a, y, s, eta, self.omega, self.L, self.adjacency_list)

  def next_infected_probabilities(self, a, eta=ETA):
    if self.contaminator is not None and self.epsilon > 0:
      current_X_at_action = self.data_block_at_action(-1, a)
      contaminator_probs = self.contaminator.predict_proba(current_X_at_action)[:, -1]
      if self.epsilon == 1.0:
        return contaminator_probs
      else:
        SIS_probs = self.infection_probability(a, self.current_infected, self.current_state, eta=eta)
        return (1 - self.epsilon) * SIS_probs + self.epsilon * contaminator_probs
    else:
      return self.infection_probability(a, self.current_infected, self.current_state, eta=eta)

  def add_infections(self, y):
    self.Y = np.vstack((self.Y, y))
    self.current_infected = y

  def next_infections(self, a):
    """
    Updates the vector indicating infections (self.current_infected).
    Computes probability of infection at each state, then generates corresponding
    Bernoullis.
    :param a: self.L-length binary array of actions at each state
    """
    super(SIS, self).next_infections(a)
    next_infected_probabilities = self.next_infected_probabilities(a)
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.true_infection_probs.append(next_infected_probabilities)
    self.add_infections(next_infections)

  def neighbor_infection_and_treatment_status(self, l, a, y):
    neighbor_ixs = self.adjacency_list[l]
    num_infected_neighbors = int(np.sum(y[neighbor_ixs]))
    num_treated_and_infected_neighbors = \
      int(np.sum(np.multiply(a[neighbor_ixs], y[neighbor_ixs])))
    num_untreated_and_infected_neighbors = num_infected_neighbors - num_treated_and_infected_neighbors
    return num_treated_and_infected_neighbors, num_untreated_and_infected_neighbors

  def update_likelihood_for_location(self, l, action, last_infections, next_infections,
                                     counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected):
    a_l = action[l]
    y_l = next_infections[l]
    num_treated_and_infected_neighbors, num_untreated_and_infected_neighbors = \
      self.neighbor_infection_and_treatment_status(l, action, last_infections)

    if y_l:
      counts_for_likelihood_next_infected[int(a_l), num_untreated_and_infected_neighbors,
                                          num_treated_and_infected_neighbors] += 1
      self.indices_for_likelihood_next_infected[int(a_l)][num_untreated_and_infected_neighbors][num_treated_and_infected_neighbors]\
        .append((self.T, l))
    else:
      counts_for_likelihood_next_not_infected[int(a_l), num_untreated_and_infected_neighbors,
                                              num_treated_and_infected_neighbors] += 1
      self.indices_for_likelihood_next_not_infected[int(a_l)][num_untreated_and_infected_neighbors][num_treated_and_infected_neighbors]\
        .append((self.T, l))
    return counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected

  def update_likelihood_information(self, action, next_infections):
    last_infections = self.Y[-2, :]
    for l in range(self.L):
      is_infected = last_infections[l]
      if not is_infected:
        self.counts_for_likelihood_next_infected, self.counts_for_likelihood_next_not_infected = \
          self.update_likelihood_for_location(l, action, last_infections, next_infections,
                                              self.counts_for_likelihood_next_infected,
                                              self.counts_for_likelihood_next_not_infected)

  def update_obs_history(self, a):
    """
    :param a: self.L-length array of binary actions at each state
    """
    super(SIS, self).update_obs_history(a)
    raw_data_block = np.column_stack((self.S_indicator[-2,:], a, self.Y[-2,:]))
    data_block = self.psi(raw_data_block)

    # Main features
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)
    self.update_likelihood_information(a, self.current_infected)

  def data_block_at_action(self, data_block_ix, action, raw=False):
    """
    Replace action in raw data_block with given action.
    """
    super(SIS, self).data_block_at_action(data_block_ix, action)
    if raw:
     new_data_block = copy.copy(self.X_raw[data_block_ix])
     new_data_block[:, 1] = action
    else:
      new_data_block = self.psi(np.column_stack((self.S_indicator[data_block_ix, :], action, self.Y[data_block_ix, :])))
    return new_data_block

  def raw_data_block_at_action(self, data_block_ix, action):
    """

    :param data_block_ix:
    :param action:
    :return:
    """
    new_raw_data_block = copy.copy(self.X_raw[data_block_ix])
    new_raw_data_block[:, 1] = action
    return new_raw_data_block

  def compute_joint_gradient_and_hessian(self, mb_params, mf_params):
    """
    Compute the gradient and hessian of the stacked estimating equations used to estimate one-step
    mb and mf q functions.  (Assume mf_params were fit to env.X using SKLogit2.

    ToDo: This can be optimized!

    :param mb_params:
    :param mf_params:
    :return:
    """
    dim = len(mb_params) + len(mf_params)
    grad_outer = np.zeros((dim, dim))
    hess = np.zeros((dim, dim))

    for t in range(self.T):
      data_block, raw_data_block = self.X[t], self.X_raw[t]
      for l in range(self.L):
        x_raw, x, y = raw_data_block[l, :], data_block[l, :], self.y[t][l]

        # MB gradient
        if raw_data_block[l, 2]:
          # Compute gradient of  recovery model
          recovery_features = np.concatenate(([1.0], x_raw[1]))
          mb_grad = logit_gradient(recovery_features, y, mb_params[-2:])
          mb_hess = logit_hessian(recovery_features, mb_params[-2:])
        else:
          # Compute gradient of infection model
          mb_grad = None

        # MF gradient
        mf_features = np.concatenate(([1.0], x))
        mf_grad = logit_gradient(mf_features, y, mf_params)
        mf_hess = logit_hessian(mf_features, mf_params)

        grad_lt = np.concatenate((mb_grad, mf_grad))
        grad_outer_lt = np.outer(grad_lt, grad_lt)
        hess_lt = block_diag(mb_hess, mf_hess)

        grad_outer += grad_outer_lt
        hess += hess_lt

    hess_inv = np.linalg.inv(hess_inv)
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(self.L * self.T)
    return cov


def logit_gradient(x, y, beta):
  x_dot_beta = np.dot(x, beta)
  expit_x_dot_beta = expit(x_dot_beta)
  expit_derivative_ = expit_derivative(x_dot_beta)
  success_component = y / expit_x_dot_beta * expit_derivative_ * x
  failure_component = (1 - y) / (1 - expit_x_dot_beta) * -expit_derivative_ * x
  return success_component + failure_component


def logit_hessian(x, beta):
  outer_ = np.outer(x, x)
  expit_x_dot_beta = expit(np.dot(x, beta))
  return outer_ * expit_x_dot_beta * (1 - expit_x_dot_beta)




