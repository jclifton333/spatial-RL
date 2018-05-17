import numpy as np
import pdb

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