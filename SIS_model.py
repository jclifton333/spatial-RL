'''
Implementing susceptible-infected-susceptible (SIS) models described in 
spatial QL paper.
'''

import numpy as np
from scipy.special import expit
from lookahead import lookahead, Q, Q_max
from autologit import unconditional_logit, autologit
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class SIS(object):
  # Fixed generative model parameters
  ZETA = 1
  TAU = 1 
  
  def __init__(self, adjacency_matrix, omega, sigma):
    '''
    :param adjacency_matrix: 2d binary array corresponding to network for gen model 
    :param omega: parameter in [0,1] for mixing two SIS models
    '''
    
    self.SIGMA = sigma
    self.adjacency_matrix = adjacency_matrix
    self.nS = self.adjacency_matrix.shape[0]
    self.adjacency_list = [[lprime for lprime in range(self.nS) if self.adjacency_matrix[l, lprime] == 1] for l in range(self.nS)]
    self.omega = omega
    self.S = np.zeros((1, self.nS)) 
    self.Y = np.array([np.random.binomial(n=1, p=0.3, size=self.nS)])
    self.A = np.zeros((0, self.nS))
    self.true_infection_probs = np.zeros((0, self.nS))
    self.state_covariance = self.TAU**2 * np.eye(self.nS)
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
    
  def step(self, a): 
    '''
    Move model forward according to action a. 
    :param a: self.nS-length array of binary actions at each state 
    '''
    next_state = self.next_state() 
    self.next_infections(a) 
    self.A = np.vstack((self.A, a))
    self.T += 1
    return next_state
    
#Test
#Settings
from generate_network import lattice
omega = 0.5 
L = 25
T = 10 
K = 3
sigma = np.array([-1, -3, -3, -1, -1, 1, 1]) #Tuned to have ~0.5 infections at these settings
m = lattice(L)
a = np.random.binomial(n=1, p=1, size=L)
evaluation_budget = 20
treatment_budget = 15
gamma = 0.9
feature_function = lambda x: x
mean = 0
for rep in range(20):
  g = SIS(m, omega, sigma)
  for i in range(T):
    s = g.step(a)    
    logit = lookahead(K, gamma, g, evaluation_budget, treatment_budget, RandomForestClassifier, RandomForestClassifier, feature_function)
    Q_fn_t = lambda a: Q(a, logit, g, g.A.shape[0], feature_function)
    _, a = Q_max(Q_fn_t, s, evaluation_budget, treatment_budget)
  mean += (np.sum(g.Y) - mean) / (rep + 1)
print(mean)