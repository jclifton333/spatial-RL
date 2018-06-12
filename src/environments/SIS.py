"""
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
"""

import numpy as np
from scipy.special import expit
from .SpatialDisease import SpatialDisease
from src.utils.features import get_all_paths
import pdb


class SIS(SpatialDisease):
  PATH_LENGTH = 3 # For path-based features
  # Fixed generative model parameters
  BETA_0 = 0.9
  BETA_1 = 1.0
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
          - 1)
  ETA_3 = np.log(((1 - PROB_INF * 0.75) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1)
  ETA_5 = np.log(1 / (1 - PROB_REC) - 1)
  ETA_6 = np.log(1 / ((1 - PROB_REC) * 0.5) - 1) - ETA_5
  ETA = np.array([ETA_0, ETA_3, ETA_2, ETA_3, ETA_4, ETA_5, ETA_6])
  # pdb.set_trace()

  def __init__(self, L, omega, feature_function, generate_network):
    """
    :param omega: parameter in [0,1] for mixing two SIS models
    :param generate_network: function that accepts network size L and returns adjacency matrix
    """
    adjacency_matrix = generate_network(L)
    self.list_of_path_lists = get_all_paths(adjacency_matrix, SIS.PATH_LENGTH - 1)

    SpatialDisease.__init__(self, adjacency_matrix, feature_function)
    self.omega = omega
    self.state_covariance = self.BETA_1 * np.eye(self.L)
    
    self.S = np.zeros((1, self.L))
    self.S_indicator = self.S > 0
    self.Phi = [] # Network-level features
    self.current_state = self.S[-1,:]

  def reset(self):
    """
    Reset state and observation histories.
    """
    # super.reset()
    super(SIS, self).reset()
    self.S = np.zeros((1, self.L))
    self.S_indicator = self.S > 0
    self.Phi = []
    self.current_state = self.S[-1,:]

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
    powers_of_2_matrix = np.array([[np.power(2.0, q*i-j) for j in range(q)]
                                   for i in range(k)])
    return 1 + np.sum( np.multiply(b, powers_of_2_matrix) )

  def phi_k_m(self, k, m, data_block):
    """
    :param k:
    :param m:
    :param data_block:
    :return:
    """
    return np.sum([self.m_r(r, data_block) == 1 + m for r in self.list_of_path_lists
                   if len(r) == k])

  def phi_k(self, k, data_block):
    """
    :param k: path length
    :param data_block:
    :return:
    """
    M = 3**k
    phi_k = np.zeros(M)
    for m in range(M):
      phi_k[m] = self.phi_k_m(k, m, data_block)
    return phi_k

  def phi(self, data_block):
    """
    :param data_block:
    :return:
    """
    phi = np.zeros(0)
    for k in range(1, SIS.PATH_LENGTH + 1):
      phi_k = self.phi_k(k, data_block)
      phi = np.append(phi, phi_k)
    return phi

  ##############################################################
  ##            End path-based feature function stuff         ##
  ##############################################################

  def next_state(self): 
    """
    Update state array acc to AR(1) 
    :return next_state: self.L-length array of new states 
    """
    super(SIS, self).next_state()
    next_state = np.random.multivariate_normal(mean=self.BETA_0*self.current_state, cov=self.state_covariance)
    self.S = np.vstack((self.S, next_state))
    self.S_indicator = np.vstack((self.S_indicator, next_state > 0))
    self.current_state = next_state 
    return next_state

  def next_infected_probabilities(self, a):
    z = np.random.binomial(1, self.omega)
    indicator = (z*self.current_state <= 0)
    a_times_indicator = np.multiply(a, indicator)

    infected_indices = np.where(self.current_infected > 0)
    not_infected_indices = np.where(self.current_infected == 0)

    next_infected_probabilities = np.zeros(self.L)
    next_infected_probabilities[not_infected_indices] = self.p_l(a_times_indicator, not_infected_indices)
    next_infected_probabilities[infected_indices] = 1 - self.q_l(a_times_indicator[infected_indices])

    return next_infected_probabilities

  def next_infections(self, a):
    """
    Updates the vector indicating infections (self.current_infected).
    Computes probability of infection at each state, then generates corresponding 
    Bernoullis.    
    :param a: self.L-length binary array of actions at each state     
    """
    super(SIS, self).next_infections(a)
    next_infected_probabilities = self.next_infected_probabilities(a)

    # For debugging, get infection probs in the case that all locations are treated
    # all_treated_probs = np.zeros(self.L)
    # all_treated_probs[not_infected_indices] = self.p_l(np.zeros(self.L), not_infected_indices)
    # all_treated_probs[infected_indices] = 1 - self.q_l(np.zeros(self.L)[infected_indices])
    # print(all_treated_probs - next_infected_probabilities)
    # pdb.set_trace()

    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.R = np.append(self.R, np.sum(next_infections))
    self.true_infection_probs.append(next_infected_probabilities)
    self.current_infected = next_infections
  
  ##############################################################
  ## Infection probability helper functions (see draft p. 13) ##
  ##############################################################
  
  def p_l0(self, a_times_indicator):
    logit_p_0 = self.ETA[0] + self.ETA[1] * a_times_indicator
    p_0 = expit(logit_p_0)
    return p_0 
    
  def q_l(self, a_times_indicator):
    logit_q = self.ETA[5] + self.ETA[6] * a_times_indicator
    q = expit(logit_q)
    return q 
    
  def one_minus_p_llprime(self, a_times_indicator, indices): 
    product_vector = np.array([])
    for l in indices[0].tolist():
      # Get infected neighbors
      infected_neighbor_indices = np.intersect1d(self.adjacency_list[l], indices)
      a_times_indicator_lprime = a_times_indicator[infected_neighbor_indices]
      logit_p_l = self.ETA[2] + self.ETA[3]*a_times_indicator[l] + \
                  self.ETA[4]*a_times_indicator_lprime
      p_l = expit(logit_p_l)
      product_l = np.product(1 - p_l)
      product_vector = np.append(product_vector, product_l) 
    return product_vector 
    
  def p_l(self, a_times_indicator, indices): 
    p_l0 = self.p_l0(a_times_indicator[indices])
    one_minus_p_llprime = self.one_minus_p_llprime(a_times_indicator, indices)
    product = np.multiply(1 - p_l0, one_minus_p_llprime) 
    return 1 - product 
        
  ################################################
  ## End infection probability helper functions ##
  ################################################
    
  def neighborFeatures(self, data_block):
    """
    For each location in data_block, compute neighbor feature vector
      [sum of positive(s), sum of s*a, sum of a, sum of y, sum of a * y]
    """
    neighborFeatures = np.zeros((0, 5))
    for l in range(self.L):
      S_neighbor, A_neighbor, Y_neighbor = data_block[self.adjacency_list[l],:].T
      neighborFeatures_l = np.array([np.sum(np.clip(S_neighbor, a_min=0, a_max=None)), np.sum(np.multiply(S_neighbor, A_neighbor)),
                                     np.sum(A_neighbor), np.sum(np.multiply(A_neighbor, Y_neighbor)), np.sum(Y_neighbor)])
      neighborFeatures = np.vstack((neighborFeatures, neighborFeatures_l))

    return neighborFeatures    
  
  def data_block_at_action(self, data_block, action):
    """
    Replace action in raw data_block with given action.
    """
    assert data_block.shape[1] == 3  
    new_data_block = np.column_stack((data_block[:,0], action, data_block[:,2]))
    features = self.neighborFeatures(new_data_block)
    new_data_block = np.column_stack((features, self.featureFunction(new_data_block)))
    return new_data_block

  def updateObsHistory(self, a):
    """
    :param a: self.L-length array of binary actions at each state
    """
    super(SIS, self).updateObsHistory(a)
    # Get location-level features
    raw_data_block = np.column_stack((self.S[-2,:], a, self.Y[-2,:]))
    neighborFeatures = self.neighborFeatures(raw_data_block)
    data_block = np.column_stack((neighborFeatures, self.featureFunction(raw_data_block)))
    self.X_raw.append(raw_data_block)
    self.X.append(data_block)
    self.y.append(self.current_infected)

    # Get network-level features
    raw_data_block[:, 0] = self.S_indicator[-2, :]
    self.Phi.append(self.phi(raw_data_block))

  def data_block_at_action(self, data_block, action):
    """
    Replace action in raw data_block with given action.
    """
    super(SIS, self).data_block_at_action(data_block, action)
    assert data_block.shape[1] == 3
    new_data_block = np.column_stack((data_block[:, 0], action, data_block[:, 2]))
    features = self.neighborFeatures(new_data_block)
    new_data_block = np.column_stack((features, self.featureFunction(new_data_block)))
    return new_data_block
