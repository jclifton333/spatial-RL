'''
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
'''

import numpy as np
from scipy.special import expit

class SIS(object):
  # Fixed generative model parameters
  ZETA = 1
  TAU = 0.1
  INITIAL_INFECT_PROB = 0.3
  
  def __init__(self, adjacency_matrix, omega, sigma, featureFunction):
    '''
    :param adjacency_matrix: 2d binary array corresponding to network for gen model 
    :param omega: parameter in [0,1] for mixing two SIS models
    '''
    
    self.featureFunction = featureFunction
    
    #Generative model parameters
    self.nS = adjacency_matrix.shape[0]
    self.SIGMA = sigma
    self.omega = omega
    self.state_covariance = self.TAU**2 * np.eye(self.nS)
    
    #Adjacency info    
    self.adjacency_matrix = adjacency_matrix
    self.adjacency_list = [[lprime for lprime in range(self.nS) if self.adjacency_matrix[l, lprime] == 1] for l in range(self.nS)]
    
    #Observation history
    self.S = np.array([np.random.multivariate_normal(mean=np.zeros(self.nS), cov=self.state_covariance)])
    self.Y = np.array([np.random.binomial(n=1, p=self.INITIAL_INFECT_PROB, size=self.nS)])
    self.A = np.zeros((0, self.nS))
    self.R = np.array([np.sum(self.Y[-1,:])])
    self.X = [] #Will hold blocks [S_t, A_t, Y_t] each each time t
    self.y = [] #Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = np.zeros((0, self.nS))
    
    #Current network status
    self.current_state = self.S[-1,:]
    self.current_infected = self.Y[-1,:]
    self.T = 0
    
  def reset(self):
    '''
    Reset state and observation histories.
    '''
    #Observation history
    self.S = np.array([np.random.multivariate_normal(mean=np.zeros(self.nS), cov=self.state_covariance)])
    self.Y = np.array([np.random.binomial(n=1, p=self.INITIAL_INFECT_PROB, size=self.nS)])
    self.A = np.zeros((0, self.nS))
    self.R = np.array([np.sum(self.Y[-1,:])])
    self.X = [] #Will hold blocks [S_t, A_t, Y_t] each each time t
    self.y = [] #Will hold blocks [Y_tp1] for each time t
    self.true_infection_probs = np.zeros((0, self.nS))
    
    #Current network status
    self.current_state = self.S[-1,:]
    self.current_infected = self.Y[-1,:]
    self.T = 0

    
  def next_state(self): 
    '''
    Update state array acc to AR(1) 
    :return next_state: self.nS-length array of new states 
    '''
    next_state = np.random.multivariate_normal(mean=self.ZETA*self.current_state, cov=self.state_covariance)
    self.S = np.vstack((self.S, next_state))
    self.current_state = next_state 
    return next_state  
    
  def next_infections(self, a): 
    '''
    Updates the vector indicating infections (self.current_infected).
    Computes probability of infection at each state, then generates corresponding 
    Bernoullis.    
    :param a: self.nS-length binary array of actions at each state     
    '''
    z = np.random.binomial(1, self.omega) 
    indicator = (z*self.current_state <= 0) 
    a_times_indicator = np.multiply(a, indicator)
    
    infected_indices = np.where(self.current_infected > 0)
    not_infected_indices = np.where(self.current_infected == 0)

    next_infected_probabilities = np.zeros(self.nS)
    next_infected_probabilities[not_infected_indices] = self.p_l(a_times_indicator, not_infected_indices)
    next_infected_probabilities[infected_indices] = 1 - self.q_l(a_times_indicator[infected_indices]) 
    next_infections = np.random.binomial(n=[1]*self.nS, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.R = np.append(self.R, np.sum(next_infections))
    self.true_infection_probs = np.vstack((self.true_infection_probs, next_infected_probabilities))
    self.current_infected = next_infections
  
  ##############################################################
  ## Infection probability helper functions (see draft p. 13) ##
  ##############################################################
  
  def p_l0(self, a_times_indicator):
    logit_p_0 = self.SIGMA[0] + self.SIGMA[1] * a_times_indicator
    p_0 = expit(logit_p_0)
    return p_0 
    
  def q_l(self, a_times_indicator):
    logit_q = self.SIGMA[5] + self.SIGMA[6] * a_times_indicator 
    q = expit(logit_q)
    return q 
    
  def one_minus_p_llprime(self, a_times_indicator, indices): 
    product_vector = np.array([])
    for l in indices[0].tolist(): 
      a_times_indicator_lprime = a_times_indicator[self.adjacency_list[l]]
      logit_p_l = self.SIGMA[2] + self.SIGMA[3]*a_times_indicator[l] + \
                  self.SIGMA[4]*a_times_indicator_lprime 
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
    '''
    For each location in data_block, compute neighbor feature vector
      [sum of positive(s), sum of s*a, sum of a, sum of y, sum of a * y]
    '''
    neighborFeatures = np.zeros((0, 5))
    for l in range(self.nS):
      S_neighbor, A_neighbor, Y_neighbor = data_block[self.adjacency_list[l],:].T
      neighborFeatures_l = np.array([np.sum(np.clip(S_neighbor, a_min=0, a_max=None)), np.sum(np.times(S_neighbor, A_neighbor)),
                                     np.sum(A_neighbor), np.sum(np.times(A_neighbor, Y_neighbor)), np.sum(Y_neighbor)])      
      neighborFeatures = np.vstack((neighborFeatures, neighborFeatures_l))
    return neighborFeatures      
    
  def updateObsHistory(self, a):
    '''
    :param a: self.nS-length array of binary actions at each state
    '''
    data_block = np.column_stack((self.S[-2,:], a, self.Y[-2,:]))
    neighborFeatures = self.neighborFeatures(data_block)
    data_block = np.column_stack((neighborFeatures, self.featureFunction(data_block)))
    self.X.append(data_block)
    self.y.append(self.current_infected)    
  
  def step(self, a): 
    '''
    Move model forward according to action a. 
    :param a: self.nS-length array of binary actions at each state 
    '''
    self.A = np.vstack((self.A, a))
    next_state = self.next_state() 
    self.next_infections(a) 
    self.updateObsHistory(a)
    self.T += 1
    return next_state
    

    
    
    