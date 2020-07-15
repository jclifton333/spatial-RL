# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import numpy as np
from itertools import permutations
from src.environments.generate_network import lattice
from scipy.optimize import minimize
import pdb


def embed_location(X_raw, neighbors_list, g, h, l, J):
  """
  Embed raw features at location l using functions g and h, following notation in current draft (July 2020)
  of the spatial Q-learning paper.

  """
  x_l = X_raw[l, :]
  neighbors_list_l = [l] + neighbors_list[l]  # ToDo: check that neighbors_list doesn't include l
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
        x_l1 = X_raw[perm[0], :]
        x_list = X_raw[perm[1:], :]
        fkm1_val = fk(x_list, k-1)
        h_val = h(x_l1)
        g_val = g(h_val, fkm1_val)
        result += g_val[0] / len(permutations_k)
      return result

  E_l = fk(X_raw[neighbors_list_l, :], N_l)
  return E_l


def embed_network(X_raw, adjacency_list, g, h, J):
  E = np.zeros((0, J))
  for l in range(X_raw.shape[0]):
    E_l = embed_location(X_raw, adjacency_list, g, h, l, J)
    E = np.vstack((E, E_l))
  return E


def learn_one_dimensional_h(X, y, adjacency_list, g, h, J):
  p = X.shape[1]

  def loss(h_param):
    h_given_param = lambda a: h(a, h_param)
    E = embed_network(X, adjacency_list, g, h_given_param, J)
    y_hat = E[:, 0]
    return np.sum((y - y_hat)**2)

  res = minimize(loss, x0=np.ones(p), method='L-BFGS-B')
  return res.x


if __name__ == "__main__":
  # Test
  adjacency_mat = lattice(16)
  adjacency_list = [[j for j in range(16) if adjacency_mat[i, j]] for i in range(16)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  X = np.random.normal(size=(16, 2))
  y = np.random.normal(loc=np.array([np.sum(X[adjacency_list[l]]) for l in range(16)]))

  g = lambda a, b: a + b
  h = lambda a, h_param: [np.dot(a, h_param)]
  J = 1

  estimated_h_param = learn_one_dimensional_h(X, y, adjacency_list, g, h, J)
  print(estimated_h_param)


















