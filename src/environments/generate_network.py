import numpy as np
import pdb

'''
Functions for generating adjacency matrices for networks to be used in SpatialDisease
sims.
'''

def lattice(size):
  '''
  Return adjacency matrix for sqrt(size) x sqrt(size) lattice.
  '''
  dim = int(np.floor(np.sqrt(size)))
  adjacency_matrix = np.zeros((size, size))
  for i in range(size): 
    for j in range(size): 
      if (j == i + 1) and ((i + 1) % dim != 0): 
        adjacency_matrix[i, j] = 1 
      elif (j == i - 1) and (i % dim != 0):
        adjacency_matrix[i, j] = 1 
      elif (j == i + dim) and (i + 1 + dim <= size):
        adjacency_matrix[i, j] = 1 
      elif (j == i - dim) and (i + 1 - dim > 0):
        adjacency_matrix[i, j] = 1
  return adjacency_matrix 

def Barabasi_Albert(size):
  '''
  Random preferential attachment model
  See https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model#Algorithm
  '''
  PROP_INIT = 0.2 #Initial graph size will be PROP_INIT * size
  P = 0.5 #Probability of uniform random connection (otherwise, preferentially attach)
  
  #Initialize graph
  initial_graph_size = int(np.floor(PROP_INIT * size))
  assert initial_graph_size > 0
  
  #Start with fully connected adjacency matrix  
  adjacency_matrix = np.zeros((size, size))
  for l in range(initial_graph_size):
    for l_prime in range(initial_graph_size):
      adjacency_matrix[l, l_prime] = 1
      
  for l in range(initial_graph_size, size):
    #Randomly attach
    if np.random.random() < P:
      l_prime = np.random.choice(l)
    else:
      degrees = np.sum(adjacency_matrix[:l,], axis=1)
      probs   = degrees / np.sum(degrees)
      l_prime = np.random.choice(l, p=probs)
    adjacency_matrix[l, l_prime] = 1
    adjacency_matrix[l_prime, l] = 1
    
  return adjacency_matrix

  
  