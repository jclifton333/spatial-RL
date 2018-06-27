import numpy as np
import networkx as nx
import pdb

'''
Functions for generating adjacency matrices for networks to be used in SpatialDisease
sims.
'''


def pseudo_square_root(integer):
  """
  Stupid search for pseudo square root (largest factor that doesn't exceed sqrt(integer).)
  """
  assert (integer < 1e6, "Number too big, choose something less than 1e6.")
  sqrt = np.sqrt(integer)
  psr = 1
  psr_complement = integer
  i = 2
  while i <= sqrt:
    if integer % i == 0:
      psr = i
      psr_complement = integer / i
    i += 1
  return psr, psr_complement


def lattice(size):
  """
  Return adjacency matrix for sqrt(size) x sqrt(size) lattice.
  :param size:
  :return:
  """
  nrow, ncol = pseudo_square_root(size)
  adjacency_matrix = np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      if (j == i + 1) and ((i + 1) % ncol != 0):
        adjacency_matrix[i, j] = 1
      elif (j == i - 1) and (i % ncol != 0):
        adjacency_matrix[i, j] = 1
      elif (j == i + ncol) and (i + 1 + nrow <= size):
        adjacency_matrix[i, j] = 1
      elif (j == i - ncol) and (i + 1 - nrow > 0):
        adjacency_matrix[i, j] = 1
  return adjacency_matrix


def Barabasi_Albert(size):
  """
  Random preferential attachment model
  See https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model#Algorithm
  """
  PROP_INIT = 0.2  # Initial graph size will be PROP_INIT * size
  P = 0.5  # Probability of uniform random connection (otherwise, preferentially attach)

  # Initialize graph
  initial_graph_size = int(np.floor(PROP_INIT * size))
  assert initial_graph_size > 0

  # Start with fully connected adjacency matrix
  adjacency_matrix = np.zeros((size, size))
  for l in range(initial_graph_size):
    for l_prime in range(initial_graph_size):
      adjacency_matrix[l, l_prime] = 1

  for l in range(initial_graph_size, size):
    # Randomly attach
    if np.random.random() < P:
      l_prime = np.random.choice(l)
    else:
      degrees = np.sum(adjacency_matrix[:l, ], axis=1)
      probs = degrees / np.sum(degrees)
      l_prime = np.random.choice(l, p=probs)
    adjacency_matrix[l, l_prime] = 1
    adjacency_matrix[l_prime, l] = 1

  return adjacency_matrix


def order_stat(arr, k):
  return np.partition(arr, k)[:k].max()

def argmin_2d(arr):
  return np.unravel_index(arr.argmin(), arr.shape)


def random_nearest_neighbor(size):
  # Described in RSS-Supplement
  k = 3
  C = np.random.uniform(size=(size, 2))
  E = np.array([[np.linalg.norm(C[l, :] - C[lprime, :]) for lprime in range(size)]
                for l in range(size)])
  omega_tilde = np.zeros((size, size))
  for l in range(size):
    E_l = E[l, :]
    E_l_upper = order_stat(E_l, k + 1)
    Nk_l = np.where(E_l <= E_l_upper)
    omega_tilde[l, Nk_l] = 1
  omega_tilde_nx = nx.from_numpy_matrix(omega_tilde)
  B = nx.connected_components(omega_tilde_nx)
  while len(list(B)) > 1:
    best_pair = None
    best_dist = float('inf')
    for i in range(len(list(B))):
      for j in range(i + 1, len(B)):
        B_i = B[i]
        B_j = B[j]
        distances = np.array([[E[l, lprime] for l in B_i]
                              for lprime in B_j])
        min_dist = np.min(distances)
        if best_pair is None or min_dist < best_dist:
          best_pair_ixs = argmin_2d(distances)
          best_pair = B_i[best_pair_ixs[1]], B_j[best_pair_ixs[0]]
          best_dist = min_dist
    omega_tilde_nx.add_edge(best_pair[0], best_pair[1])
    B = nx.connected_components(omega_tilde_nx)
  return nx.adjacency_matrix(omega_tilde_nx)

