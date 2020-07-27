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
import torch.nn as nn
from pygcn.utils import accuracy
from pygcn.models import GCN
import pdb


class GGCN(nn.Module):
  """
  Generalized graph convolutional network (not sure yet if it's a generalization strictly speaking).
  """
  def __init__(self, nfeat, J, neighbor_subset_limit=2, samples_per_k=None):
    super(GGCN, self).__init__()
    if neighbor_subset_limit > 1:
      self.g1 = nn.Linear(2*J, 100)
      self.g2 = nn.Linear(100, J)
    self.h1 = nn.Linear(nfeat, 100)
    self.h2 = nn.Linear(100, J)
    self.final = nn.Linear(J + nfeat*(neighbor_subset_limit > 1), 1)
    self.neighbor_subset_limit = neighbor_subset_limit
    self.J = J
    self.samples_per_k = samples_per_k

  def h(self, b):
    b = self.h1(b)
    b = self.h2(b)
    return b

  def g(self, bvec):
    bvec = self.g1(bvec)
    bvec = F.relu(bvec)
    bvec = self.g2(bvec)
    bvec = F.relu(bvec)
    return bvec

  def forward(self, X_, adjacency_lst):
    L = X_.shape[0]
    final_ = torch.tensor([])
    X_ = torch.tensor(X_).float()
    for l in range(L):
      neighbors_l = adjacency_lst[l] + [l]
      N_l = np.min((len(neighbors_l), self.neighbor_subset_limit))

      def fk(b, k):
        permutations_k = list(permutations(neighbors_l, int(k)))
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        if k == 1:
          return self.h(b[0])
        else:
          result = torch.zeros(self.J)
          for perm in permutations_k:
            x_l1 = torch.tensor(X_[perm[0], :])
            x_list = X_[perm[1:], :]
            fkm1_val = fk(x_list, k - 1)
            h_val = self.h(x_l1)
            h_val_cat_fkm1_val = torch.cat((h_val, fkm1_val))
            g_val = self.g(h_val_cat_fkm1_val)
            result += g_val / len(permutations_k)
          return result

      if N_l > 1:
        x_l = X_[l, :]
        E_l = torch.cat((x_l, fk(X_[neighbors_l, :], N_l)))
      else:
        E_l = fk([X_[l, :]], N_l)
      final_l = self.final(E_l)
      final_ = torch.cat((final_, final_l))

    params = list(self.parameters())
    yhat = F.sigmoid(final_)
    return yhat


def learn_ggcn(X_list, y_list, adjacency_list, n_epoch=200, nhid=10, batch_size=5, verbose=False,
               neighbor_subset_limit=2, samples_per_k=None):
  # See here: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
  # Specify model
  p = X_list[0].shape[1]
  T = len(X_list)
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.FloatTensor(y) for y in y_list]
  model = GGCN(nfeat=p, J=nhid, neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k)
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  # Train
  for epoch in range(n_epoch):
    avg_acc_train = 0.
    batch_ixs = np.random.choice(T, size=batch_size)
    X_batch = [X_list[ix] for ix in batch_ixs]
    y_batch = [y_list[ix] for ix in batch_ixs]
    for X, y in zip(X_batch, y_batch):
      model.train()
      optimizer.zero_grad()
      output = model(X, adjacency_list)
      criterion = nn.BCELoss()
      loss_train = criterion(output, y)
      acc = ((output > 0.5) == y).float().mean()
      loss_train.backward()
      optimizer.step()
      avg_acc_train += acc / batch_size

    # Tracking diagnostics
    grad_norm = np.sqrt(np.sum([param.grad.data.norm(2).item()**2 for param in model.parameters()]))
    yhat0 = model(X_list[0], adjacency_list)

    if verbose:
      print('Epoch: {:04d}'.format(epoch+1),
            'acc_train: {:.4f}'.format(avg_acc_train),
            'grad_norm: {:.4f}'.format(grad_norm))

  if verbose:
    final_acc_train = 0.
    for X, y in zip(X_list, y_list):
      output = model(X, adjacency_list)
      acc = ((output > 0.5) == y).float().mean()
      final_acc_train += acc / T
    print('final_acc_train: {:.4f}'.format(final_acc_train))


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
  L, p = X_list[0].shape
  T = len(X_list)
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.LongTensor(y) for y in y_list]
  adjacency_mat += np.eye(L)
  adjacency_mat = torch.FloatTensor(adjacency_mat)
  model = GCN(nfeat=p, nhid=nhid, nclass=2, dropout=0.5)
  optimizer = optim.Adam(model.parameters(), lr=0.01)

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


















