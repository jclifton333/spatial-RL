"""
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
"""

import copy
import numpy as np
from scipy.special import expit
from .SpatialDisease import SpatialDisease
from .sis_contaminator import random_contamination
from ..utils.features import get_all_paths
from ..utils.misc import KerasLogit
import pdb
import pickle as pkl
import numba as nb
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
  # CONTAMINATOR = KerasLogit()
  # CONTAMINATOR.set_weights(CONTAMINATION_MODEL_PARAMETER, 90)

  def __init__(self, L, omega, generate_network, add_neighbor_sums=False, adjacency_matrix=None,
               initial_infections=None, initial_state=None, eta=None, beta=None,
               epsilon=0, contaminator=None):
    """
    :param omega: parameter in [0,1] for mixing two SIS models
    :param generate_network: function that accepts network size L and returns adjacency matrix
    """
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

  ##############################################################
  ##            Feature function computation                  ##
  ##############################################################

  def phi_at_location(self, l, data_block):
    phi_l = np.zeros(16)

    # Get encoding for location l
    row_l = data_block[l, :]
    ix = int(np.dot(row_l, SIS.ENCODING_MATRIX))
    phi_l[ix] = 1

    # Get encodings for l's neighbors
    for lprime in self.adjacency_list[l]:
      row_lprime = data_block[lprime, :]
      ix = int(np.dot(row_lprime, SIS.ENCODING_MATRIX))
      phi_l[8 + ix] += 1

    return phi_l

  def phi(self, data_block):
    """
    :param data_block:
    :return:
    """
    data_block[:, -1] = 1 - data_block[:, -1]
    phi = np.zeros((0, 16))
    for l in range(self.L):
      phi_l = self.phi_at_location(l, data_block)
      phi = np.vstack((phi, phi_l))
    return phi

  @staticmethod
  def is_any_element_in_set(list_, set_):
    for l in list_:
      if l in set_:
        return True
    return False

  def phi_at_action(self, old_raw_data_block, old_data_block, old_action, action):
    new_data_block = copy.copy(old_data_block)
    if self.add_neighbor_sums:
      new_data_block = new_data_block[:, :int(new_data_block.shape[1] / 2)]
    locations_with_changed_actions = set(np.where(old_action != action)[0])

    for l in range(self.L):
      l_and_neighbors = [l] + self.adjacency_list[l]
      if self.is_any_element_in_set(l_and_neighbors, locations_with_changed_actions):
        new_data_block[l, :] = self.phi_at_location(l, old_raw_data_block)
    if self.add_neighbor_sums:
      new_data_block = self.phi_neighbor(new_data_block)
    return new_data_block

  def phi_neighbor(self, data_block):
    """
    To phi features, concatenate another block of features which are the _sums of the neighboring phi features_.
    :param data_block:
    :return:
    """
    neighbor_data_block = np.zeros(data_block.shape)
    for l in range(self.L):
      neighbor_data_block[l] = np.sum(data_block[self.adjacency_list[l],:], axis=0)
    return np.column_stack((data_block, neighbor_data_block))

  def get_weighted_eigenvector_centrality(self, estimated_probabilities):
    weighted_adjacency_matrix = np.zeros((self.L, self.L))
    for l in range(self.L):
      for l_prime in self.adjacency_list[l]:
        weighted_adjacency_matrix[l, l_prime] = np.prod(estimated_probabilities[[l, l_prime]])
    g = nx.from_numpy_matrix(weighted_adjacency_matrix)
    centrality = nx.eigenvector_centrality(g)
    return np.array([centrality[node] for node in centrality])

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
    z = np.random.binomial(1, self.omega)
    indicator = (z*s <= 0)
    a_times_indicator = np.multiply(a, indicator)

    infected_indices = np.where(y > 0)
    not_infected_indices = np.where(y == 0)

    infected_probabilities = np.zeros(self.L)
    infected_probabilities[not_infected_indices] = self.p_l(a_times_indicator, not_infected_indices, infected_indices,
                                                            eta)
    infected_probabilities[infected_indices] = 1 - self.q_l(a_times_indicator[infected_indices], eta)
    return infected_probabilities

  def next_infected_probabilities(self, a, eta=ETA):
    if self.contaminator is not None and self.epsilon > 0:
      current_X_at_action = self.data_block_at_action(-1, a)
      # contaminator_probs = self.contaminator.predict_proba(current_X_at_action)[:, -1]
      contaminator_probs = random_contamination(current_X_at_action)
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

  ##############################################################
  ## Infection probability helper functions (see draft p. 13) ##
  ##############################################################

  def p_l0(self, a_times_indicator, eta):
    logit_p_0 = eta[0] + eta[1] * a_times_indicator
    p_0 = expit(logit_p_0)
    return p_0

  def q_l(self, a_times_indicator, eta):
    logit_q = eta[5] + eta[6] * a_times_indicator
    q = expit(logit_q)
    return q

  def one_minus_p_llprime(self, a_times_indicator, not_infected_indices, infected_indices, eta):
    product_vector = np.array([])
    for l in not_infected_indices[0].tolist():
      # Get infected neighbors
      infected_neighbor_indices = np.intersect1d(self.adjacency_list[l], infected_indices)
      a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
      logit_p_l = eta[2] + eta[3]*a_times_indicator[l] + eta[4]*a_times_indicator_lprime
      p_l = expit(logit_p_l)
      product_l = np.product(1 - p_l)
      product_vector = np.append(product_vector, product_l)
    return product_vector

  def p_l(self, a_times_indicator, not_infected_indices, infected_indices, eta):
    p_l0 = self.p_l0(a_times_indicator[not_infected_indices], eta)
    one_minus_p_llprime = self.one_minus_p_llprime(a_times_indicator, not_infected_indices, infected_indices, eta)
    product = np.multiply(1 - p_l0, one_minus_p_llprime)
    return 1 - product

  ################################################
  ## End infection probability helper functions ##
  ################################################

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
    else:
      counts_for_likelihood_next_not_infected[int(a_l), num_untreated_and_infected_neighbors,
                                              num_treated_and_infected_neighbors] += 1
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

  def get_likelihood_information_for_cv_split(self, ixs):
    """
    :param ixs: List of lists containing integers in (0, self.L-1), indexing locations at each time point to keep in
                split.
    :return:
    """
    counts_for_cv_likelihood_next_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    counts_for_cv_likelihood_next_not_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    for t in range(len(ixs)):
      ix = ixs[t]
      y_next = self.y[t]
      y_current = self.Y[t, :]
      action = self.A[t, :]
      for l in ix:
        is_infected = y_current[l]
        if is_infected:
          counts_for_cv_likelihood_next_infected, counts_for_cv_likelihood_next_not_infected = \
            self.update_likelihood_for_location(l, action, y_current, y_next, counts_for_cv_likelihood_next_infected,
                                                counts_for_cv_likelihood_next_not_infected)
    return counts_for_cv_likelihood_next_infected, counts_for_cv_likelihood_next_not_infected

  def update_obs_history(self, a):
    """
    :param a: self.L-length array of binary actions at each state
    """
    super(SIS, self).update_obs_history(a)
    raw_data_block = np.column_stack((self.S_indicator[-2,:], a, self.Y[-2,:]))
    data_block = self.phi(raw_data_block)
    # Main features
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)

    self.update_likelihood_information(a, self.current_infected)

  def data_block_at_action(self, data_block_ix, action):
    """
    Replace action in raw data_block with given action.
    """
    super(SIS, self).data_block_at_action(data_block_ix, action)
    if len(self.X_raw) == 0:
      new_data_block = self.phi(np.column_stack((self.S_indicator[-1,:], action, self.Y[-1,:])))
    else:
      new_data_block = self.phi_at_action(self.X_raw[data_block_ix], self.X[data_block_ix], self.A[-1, :], action)
    return new_data_block

