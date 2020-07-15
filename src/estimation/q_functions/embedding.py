# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import numpy as np
from itertools import permutations
from src.environments.generate_network import lattice


def embed_location(X_raw, neighbors_list, g, h, l, J):
  """
  Embed raw features at location l using functions g and h, following notation in current draft (July 2020)
  of the spatial Q-learning paper.

  """
  x_l = X_raw[l, :]
  neighbors_list_l = [l] + neighbors_list[l] # ToDo: check that neighbors_list doesn't include l
  N_l = len(neighbors_list[l])

  f1 = lambda b: h(b)

  def fk(b, k):
    # ToDo: allow sampling
    permutations_k = list(permutations(neighbors_list_l, k))
    if k == 1:
      return f1(b)
    else:
      result = np.zeros(J)
      for perm in permutations_k:
        x_l1 = X_raw[perm[0]]
        x_list = X_raw[perm[1:]]
        fkm1_val = fk(x_list, k-1)
        h_val = h(x_l1)
        g_val = g(h_val, fkm1_val)
        result += g_val / len(permutations_k)
      return result

  E_l = fk(X_raw[neighbors_list_l, :], N_l)
  return E_l


if __name__ == "__main__":
  # Test embedding function
  adjacency_mat = lattice(16)
  adjacency_list = [[j for j in range(16) if adjacency_mat[i, j]] for i in range(16)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  X = np.random.poisson(lam=neighbor_counts, size=(2, 16)).T

  g = lambda x, y: x + y
  h = lambda x: x[:2]
  J = 2
  l = 1

  embed_location(X, adjacency_list, g, h, l, J)



















