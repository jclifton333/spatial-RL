'''
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
'''

import numpy as np
from scipy.special import expit

class SIS(object):
  #Fixed generative model parameters
  ZETA = 1
  TAU = 1 
  SIGMA = np.ones(7)
  
  def __init__(self, adjacency_matrix, omega):
    '''
    :param adjacency_matrix: 2d binary array corresponding to network for gen model 
    :param omega: parameter in [0,1] for mixing two SIS models
    '''
    
    self.adjacency_matrix = adjacency_matrix
    self.nS = self.adjacency_matrix.shape[0]
    self.adjaceny_list = [np.where(self.adjacency_matrix[l,:] > 0 for l in range(self.nS))]
    self.omega = omega
    self.S = np.zeros((1, self.nS)) 
    self.Y = np.zeros((1, self.nS))
    self.A = np.zeros((0, self.nS))
    self.state_covariance = self.TAU**2 * np.eye(nS)
    self.current_state = self.S[-1,:]
    self.current_infected = self.Y[-1,:]
    
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
    
    not_infected_probabilities = self.not_infected_probabilities(a_times_indicator, not_infected_indices)
    infected_probabilities = self.infected_probabilities(a_times_indicator, infected_indices)
    
    next_infected_probabilities = np.zeros(self.nS)
    next_infected_probabilities[infected_indices] = infected_probabilities
    next_infected_probabilities[not_infected_indices] = not_infected_probabilities 
    
    next_infections = np.random.binomial(n=[1]*self.nS, p=next_infected_probabilities)
    self.Y = np.vstack((self.Y, next_infections))
    self.current_infected = next_infections
  
  ##############################################################
  ## Infection probability helper functions (see draft p. 13) ##
  ##############################################################
  
  def p_l0(self, a_times_indicator):
    logit_p_0 = self.SIGMA[0] + self.SIGMA[1] * a_times_indicator
    p_0 = expit(logit_p_0)
    return p_0 
    
  def q_l(self, a_times_indicator)
    logit_q = self.SIGMA[5] + self.SIGMA[6] * a_times_indicator 
    q = expit(logit_q)
    return q 
    
  def one_minus_p_llprime(self, a_times_indicator, indices): 
    product_vector = np.array([])
    for l in indices: 
      a_times_indicator_lprime = a_times_indicator[self.adjacency_list[l]]
      logit_p_l = self.SIGMA[2] + self.SIGMA[3]*a_times_indicator[l] + \
                  self.SIGMA[4]*a_times_indicator_lprime 
      p_l = expit(logit_p_l)
      product_l = np.product(1 - p_l)
      product_vector = np.append(product_vector, product_l) 
    return product_vector 
    
  def p_l(self, a_times_indicator): 
    p_l0 = self.p_l0(a_times_indicator)
    one_minus_p_llprime = self.one_minus_p_llprime(a_times_indicator)
    product = np.multiply(1 - p_l0, one_minus_p_llprime) 
    return 1 - product 
    
  def not_infected_probabilities(self, a_times_indicator, not_infected_indices): 
    p_l = self.p_l(a_times_indicator[not_infected_indices], not_infected_indices)
    y_not_infected = self.current_infected[not_infected_indices]
    prob1 = np.power(p_1, y_not_infected)
    prob2 = np.power(1-p_1, 1-y_not_infected)
    return np.multiply(prob1, prob2)
    
  def infected_probabilities(self, a_times_indicator, infected_indices):
    q_l = self.q_l(a_times_indicator)
    y_infected = self.current_infected[infected_indices]
    prob1 = np.power(q_l, y_infected)
    prob2 = np.power(1-q_l, 1-y_infected)
    return np.multiply(prob1, prob2)
    
  ################################################
  ## End infection probability helper functions ##
  ################################################
    
  def step(self, a): 
    '''
    Move model forward according to action a. 
    :param a: self.nS-length array of binary actions at each state 
    '''
    next_state = self.next_state() 
    self.next_infections(a) 
    self.A = self.vstack((self.A, a))
    
    
