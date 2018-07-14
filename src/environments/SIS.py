"""
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
"""

import numpy as np
from scipy.special import expit
from .SpatialDisease import SpatialDisease
from ..utils.features import get_all_paths
import pdb
import numba as nb


def sum_prod(A, B):
  m, n = A.shape
  s = 1
  for i in range(m):
    for j in range(n):
      s += A[i,j]*B[i,j]
  return s


numba_sum_prod = nb.jit(nb.float64(nb.float64[:,:], nb.float64[:,:]), nopython=True)(sum_prod)


class SIS(SpatialDisease):
  PATH_LENGTH = 2 # For path-based features
  POWERS_OF_TWO_MATRICES ={
    k: np.array([[np.power(2.0, 3*i-j) for j in range(1, 3+1)] for i in range(1, k + 1)]) for k in range(1, PATH_LENGTH + 1)
  }
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

  def __init__(self, feature_function, L, omega, generate_network, adjacency_matrix=None, dict_of_path_lists=None,
               initial_infections=None, initial_state=None, eta=None, beta=None):
    """
    :param omega: parameter in [0,1] for mixing two SIS models
    :param generate_network: function that accepts network size L and returns adjacency matrix
    """
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
      self.dict_of_path_lists = get_all_paths(self.adjacency_matrix, SIS.PATH_LENGTH - 1)
    else:
      self.adjacency_matrix = adjacency_matrix
      self.dict_of_path_lists = dict_of_path_lists
    SpatialDisease.__init__(self, self.adjacency_matrix, feature_function, initial_infections)

    if initial_state is None:
      self.initial_state = np.zeros(self.L)
    else:
      self.initial_state = initial_state

    # These are for efficiently getting features at different actions
    self.map_to_path_signature = {r: None for k, r_list in self.dict_of_path_lists.items() for r in r_list}
    self.map_m_to_index_dict = {k: np.sum([9**i for i in range(1, k)]) for k in self.dict_of_path_lists.keys()}

    self.omega = omega
    self.state_covariance = self.beta[1] * np.eye(self.L)

    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = [] # Network-level features
    self.current_state = self.S[-1,:]

    # These are for efficiently computing gradients for estimating generative model
    self.max_num_neighbors = int(np.max(np.sum(self.adjacency_matrix, axis=0)))
    self.counts_for_likelihood_next_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    self.counts_for_likelihood_next_not_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))

  def reset(self):
    """
    Reset state and observation histories.
    """
    super(SIS, self).reset()
    self.map_to_path_signature = {r: None for k, r_list in self.dict_of_path_lists.items() for r in r_list}
    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = []
    self.current_state = self.S[-1,:]
    self.counts_for_likelihood_next_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))
    self.counts_for_likelihood_next_not_infected = np.zeros((2, self.max_num_neighbors + 1, self.max_num_neighbors + 1))

  ##############################################################
  ## Path-based feature function computation (see draft p7)   ##
  ##############################################################

  def get_b(self, r, data_block):
    """
    Get b vector associated with current s, y, a
    on path r.
    :param r: list of indices of states on path
    :param data_block:
    :return:
    """
    b = data_block[r,:]
    return b

  def m_r(self, r, data_block):
    """
    Compute m_r for given path as defined in paper.
    :param r: list of indices on defining path
    :param data_block:
    :return:
    """
    b = self.get_b(r, data_block)
    k, q = b.shape
    powers_of_2_matrix = SIS.POWERS_OF_TWO_MATRICES[k]
    return numba_sum_prod(b, powers_of_2_matrix)

  def phi_k(self, k, data_block):
    """
    :param k: path length
    :param data_block:
    :return:
    """
    M = 9**k
    phi_k = np.zeros((data_block.shape[0], M))
    for r in self.dict_of_path_lists[k]:
      m_r = int(self.m_r(r, data_block))
      self.map_to_path_signature[r] = m_r
      phi_k[r, [m_r - 1]*k] += 1
    return phi_k

  def phi(self, data_block):
    """
    :param data_block:
    :return:
    """
    phi = np.zeros((data_block.shape[0], 0))
    for k in range(1, SIS.PATH_LENGTH + 1):
      phi_k = self.phi_k(k, data_block)
      phi = np.column_stack((phi, phi_k))
    return phi

  # Functions for efficiently computing features-at-new-action
  def map_m_to_index(self, m, k):
    start = self.map_m_to_index_dict[k]
    return int(start + m - 1)

  def modify_m_r(self, data_block, old_action, new_action, r, k):
    old_m_r = self.map_to_path_signature[r]
    action_weights = SIS.POWERS_OF_TWO_MATRICES[k][:,1]
    m_r_diff = np.dot(action_weights, new_action[list(r)] - old_action[list(r)])
    new_m_r = old_m_r + m_r_diff
    old_ix = self.map_m_to_index(old_m_r, k)
    new_ix = self.map_m_to_index(new_m_r, k)
    data_block[r, old_ix] -= 1
    data_block[r, new_ix] += 1
    return data_block

  @staticmethod
  def is_any_element_in_set(list_, set_):
    for l in list_:
      if l in set_:
        return True
    return False

  def phi_at_action(self, data_block, old_action, action, ixs=None):
    locations_with_changed_actions = set(np.where(old_action == action)[0])
    if ixs is not None:
      # This only works when considering paths up to length 2!
      ixs_and_their_neighbors = np.append(ixs, [l for ix in ixs for l in self.adjacency_list[ix]])
      locations_with_changed_actions = locations_with_changed_actions.intersection(ixs_and_their_neighbors)

    for k, length_k_paths in self.dict_of_path_lists.items():
      for r in length_k_paths:
        if self.is_any_element_in_set(r, locations_with_changed_actions):
          data_block = self.modify_m_r(data_block, old_action, action, r, k)
    if ixs is None:
      return data_block
    else:
      return data_block[ixs,:]

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

  def next_infected_probabilities(self, a):
    z = np.random.binomial(1, self.omega)
    indicator = (z*self.current_state <= 0)
    a_times_indicator = np.multiply(a, indicator)

    infected_indices = np.where(self.current_infected > 0)
    not_infected_indices = np.where(self.current_infected == 0)

    next_infected_probabilities = np.zeros(self.L)
    next_infected_probabilities[not_infected_indices] = self.p_l(a_times_indicator, not_infected_indices,
                                                                 infected_indices)
    next_infected_probabilities[infected_indices] = 1 - self.q_l(a_times_indicator[infected_indices])

    return next_infected_probabilities

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

  def p_l0(self, a_times_indicator):
    logit_p_0 = self.eta[0] + self.eta[1] * a_times_indicator
    p_0 = expit(logit_p_0)
    return p_0

  def q_l(self, a_times_indicator):
    logit_q = self.eta[5] + self.eta[6] * a_times_indicator
    q = expit(logit_q)
    return q

  def one_minus_p_llprime(self, a_times_indicator, not_infected_indices, infected_indices):
    product_vector = np.array([])
    for l in not_infected_indices[0].tolist():
      # Get infected neighbors
      infected_neighbor_indices = np.intersect1d(self.adjacency_list[l], infected_indices)
      a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
      logit_p_l = self.eta[2] + self.eta[3]*a_times_indicator[l] + \
                  self.eta[4]*a_times_indicator_lprime
      p_l = expit(logit_p_l)
      product_l = np.product(1 - p_l)
      product_vector = np.append(product_vector, product_l)
    return product_vector

  def p_l(self, a_times_indicator, not_infected_indices, infected_indices):
    p_l0 = self.p_l0(a_times_indicator[not_infected_indices])
    one_minus_p_llprime = self.one_minus_p_llprime(a_times_indicator, not_infected_indices, infected_indices)
    product = np.multiply(1 - p_l0, one_minus_p_llprime)
    return 1 - product

  ################################################
  ## End infection probability helper functions ##
  ################################################

  def update_likelihood_for_location(self, l, action, last_infections, next_infections,
                                     counts_for_likelihood_next_infected, counts_for_likelihood_next_not_infected):
    a_l = action[l]
    y_l = next_infections[l]
    neighbor_ixs = self.adjacency_list[l]
    num_infected_neighbors = int(np.sum(last_infections[neighbor_ixs]))
    num_treated_and_infected_neighbors = \
      int(np.sum(np.multiply(action[neighbor_ixs], last_infections[neighbor_ixs])))
    num_untreated_and_infected_neighbors = num_infected_neighbors - num_treated_and_infected_neighbors

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

  def data_block_at_action(self, data_block_ix, action, ixs=None):
    """
    Replace action in raw data_block with given action.
    """
    super(SIS, self).data_block_at_action(data_block_ix, action)
    if self.A.shape[0] == 0:
      new_data_block = self.phi(np.column_stack((self.S_indicator[-1,:], action, self.Y[-1,:])))
    else:
      new_data_block = self.phi_at_action(self.X[data_block_ix], self.A[-1, :], action, ixs=ixs)
    return new_data_block

  def train_test_split(self):
    super(SIS, self).train_test_split()
    train_ixs_list = []
    test_ixs_list = []
    all_ixs = np.arange(self.L)
    n_test = int(np.floor(0.2*self.L))
    for t in range(self.T):
      test_ix = set(np.random.choice(self.L, n_test, replace=False))
      mask = np.array([(i in test_ix) for i in range(self.L)])
      test_ixs = all_ixs[mask]
      train_ixs = all_ixs[~mask]
      train_ixs_list.append(train_ixs)
      test_ixs_list.append(test_ixs)
    return train_ixs_list, test_ixs_list

  def network_features_at_action(self, data_block, action):
    """
    :param data_block:
    :param action:
    :return:
    """
    new_data_block = np.column_stack((data_block[:, 0] > 0, action, data_block[:, 2]))
    new_data_block = self.phi(new_data_block)
    return new_data_block.reshape(1,-1)
