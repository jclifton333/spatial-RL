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
from scipy.special import expit
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import accuracy
from pygcn.models import GCN
import pdb


def embed_location(X_raw, neighbors_list, g, h, l, J):
  """
  Embed raw features at location l using functions g and h, following notation in current draft (July 2020)
  of the spatial Q-learning paper.

  """
  x_l = X_raw[l, :]
  neighbors_list_l = [l] + neighbors_list[l]  # ToDo: check that neighbors_list doesn't include l
  N_l = np.min((len(neighbors_list[l]), 2))  # ToDo: restricting neighbor subset size
  f1 = lambda b: h(b)

  def fk(b, k):
    # ToDo: allow sampling
    permutations_k = list(permutations(neighbors_list_l, int(k)))
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


def learn_gcn(X_list, y_list, adjacency_mat, n_epoch=200, nhid=10, verbose=False):
  # See here: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
  # Specify model
  p = X_list[0].shape[1]
  T = len(X_list)
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.LongTensor(y) for y in y_list]
  adjacency_mat = torch.FloatTensor(adjacency_mat)
  model = GCN(nfeat=p, nhid=nhid, nclass=2, dropout=0.5)
  optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=16)

  # Train
  for epoch in range(n_epoch):
    avg_acc_train = 0.
    for X, y in zip(X_list, y_list):
      model.train()
      optimizer.zero_grad()
      output = model(X, adjacency_mat)
      loss_train = F.nll_loss(output, y)
      acc_train = accuracy(output, y)
      loss_train.backward()
      optimizer.step()
      avg_acc_train += acc_train.item() / T

    if verbose:
      print('Epoch: {:04d}'.format(epoch+1),
            'acc_train: {:.4f}'.format(avg_acc_train))

  def embedding_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    E = F.relu(model.gc1(X_, adjacency_mat)).detach().numpy()
    return E

  def model_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    y_ = model.forward(X_, adjacency_mat).detach().numpy()
    y1 = np.exp(y_[:, 1])
    return y1

  return embedding_wrapper, model_wrapper


if __name__ == "__main__":
  # Test
  adjacency_mat = lattice(16)
  adjacency_list = [[j for j in range(16) if adjacency_mat[i, j]] for i in range(16)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  n = 2
  X_list = np.array([np.random.normal(size=(16, 2)) for _ in range(2)])
  y_probs_list = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(16)]) for X in X_list])
  y_list = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in y_probs_list])
  _, predictor = learn_gcn(X_list, y_list, adjacency_mat)
  predictor(X_list[0])


















