# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)

import numpy as np
import copy
from itertools import permutations
from src.environments.generate_network import lattice, random_nearest_neighbor
from scipy.stats import pearsonr
from scipy.optimize import minimize
from scipy.special import expit
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from src.utils.misc import kl
# from pygcn.utils import accuracy
# from pygcn.models import GCN
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

import pdb


class GGCN_multi(nn.Module):
  """
  Different input layer for every number of neighbors.
  """
  def __init__(self, nfeat, J, neighbor_subset_limit=2, samples_per_k=None, recursive=False):
    # ToDo: different J for each neighbor_subset_size?

    super(GGCN, self).__init__()
    self.input_layers_1 = []
    self.input_layers_2 = []
    self.neighbor_subset_limit = neighbor_subset_limit
    for neighbor_subset_size in range(2, neighbor_subset_limit+1):
      self.input_layers_1.append(nn.Linear(nfeat*neighbor_subset_size, J))
      self.input_layers_2.append(nn.Linear(J, J))
    self.h1 = nn.Linear(J*neighbor_subset_limit, 1)
    self.samples_per_k = samples_per_k
    self.J = J

  def h(self, E):
    E = self.h1(E)
    return E

  def forward(self, X_, adjacency_lst):
    L = X_.shape[0]
    final_ = torch.tensor([])
    X_ = torch.tensor(X_).float()
    for l in range(L):
      neighbors_l = adjacency_lst[l] + [l]
      N_l = len(neighbors_l)
      full_embedding = torch.tensor([])
      for neighbor_subset_size in range(2, np.min((N_l, self.neighbor_subset_limit))):
        permutations_k = list(permutations(neighbors_l, int(neighbor_subset_size)))
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        embedding_k = torch.zeros(self.J)
        for permutation in permutations_k:
          x_permutation = X_[permutation, :]
          from torch.autograd import Variable
          x_permutation = np.hstack(x_permutation)
          embedding_permutation = self.input_layers_1[neighbor_subset_size-1](x_permutation)
          embedding_permutation = F.relu(embedding_permutation)
          embedding_permutation = self.input_layers_2[neighbor_subset_size-1](embedding_permutation)
          embedding_k += embedding_permutation / len(permutations_k)
        full_embedding = torch.cat(full_embedding, embedding_k)
    E = self.h(full_embedding)
    y = F.sigmoid(E)
    return y


class GGCN(nn.Module):
  """
  Generalized graph convolutional network (not sure yet if it's a generalization strictly speaking).
  """
  def __init__(self, nfeat, J, adjacency_lst, neighbor_subset_limit=2, samples_per_k=None, recursive=False, dropout=0.0,
               apply_sigmoid=False):
    super(GGCN, self).__init__()
    if neighbor_subset_limit > 1 and recursive:
      self.g1 = nn.Linear(2*J, J)
      # self.g2 = nn.Linear(J, J)
    self.h1 = nn.Linear(nfeat, J)
    self.h2 = nn.Linear(J, J)
    self.final1 = nn.Linear(J+nfeat, 2)
    self.dropout_final = nn.Dropout(p=dropout)
    self.neighbor_subset_limit = neighbor_subset_limit
    self.J = J
    self.samples_per_k = samples_per_k
    self.recursive = recursive
    self.apply_sigmoid = apply_sigmoid
    self.adjacency_list = adjacency_lst
    self.L = len(adjacency_lst)

  def final(self, X_, train=True):
    E = self.final1(X_)
    # E = F.sigmoid(E)
    return E

  def h(self, b):
    b = self.h1(b)
    b = F.relu(b)
    b = self.h2(b)
    b = F.relu(b)
    return b

  def g(self, bvec):
    bvec = self.g1(bvec)
    bvec = F.relu(bvec)
    # bvec = self.g2(bvec)
    # bvec = F.relu(bvec)
    return bvec

  def forward(self, X_, adjacency_lst, location_subset=None):
    if self.recursive:
      # return self.forward_recursive(X_, adjacency_lst)
      return self.forward_recursive_vec(X_, location_subset=location_subset)
    else:
      return self.forward_simple(X_, adjacency_lst)

  def forward_simple(self, X_, adjacency_lst):
    # Average a function of permutations of all neighbors, rather than all subsets of all neighbors

    L = self.L
    final_ = torch.tensor([])
    X_ = torch.tensor(X_).float()
    for l in range(L):
      neighbors_l = adjacency_lst[l] + [l]
      N_l = len(neighbors_l)

      def fk(k):
        permutations_k = list(permutations(neighbors_l, int(k)))
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        # result = torch.zeros(self.J)
        result = torch.zeros(1)
        for perm in permutations_k:
          # ToDo: just takes the first element of the permutation, makes no sense
          x_l1 = torch.tensor(X_[perm[0], :])
          h_val = self.h(x_l1)
          result += h_val / len(permutations_k)
        return result

      E_l = fk(N_l)
      # final_l = E_l
      final_l = self.final(E_l)
      final_ = torch.cat((final_, final_l))

    params = list(self.parameters())
    yhat = F.sigmoid(final_)
    return yhat

  def forward_recursive_vec(self, X_, location_subset=None, train=True):
    E = self.embed_recursive_vec(X_, locations_subset=location_subset)
    yhat = self.final(E, train=train)
    return yhat

  def sample_indices_for_recursive(self, locations_subset=None):
    L = self.L
    # Collect permutations
    self.permutations_all = {k: np.zeros((L, k, self.samples_per_k)) for k in range(2, self.neighbor_subset_limit + 1)}
    self.where_k_neighbors = {k: [] for k in range(2, self.neighbor_subset_limit + 1)}
    for l in range(L):
      neighbors_l = np.append(self.adjacency_list[l], [l])
      N_l = np.min((len(neighbors_l), self.neighbor_subset_limit))
      for k in range(2, N_l + 1):
        permutations_k = list(permutations(neighbors_l, int(k)))
        self.where_k_neighbors[k].append(l)
        if self.samples_per_k is not None:
          permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
          permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
        self.permutations_all[k][l, :] = np.array(permutations_k).T

  def embed_recursive_vec(self, X_, locations_subset=None):
    L = X_.shape[0]
    self.sample_indices_for_recursive()
    # X_ = torch.tensor(X_)

    # Collect permutations
    # permutations_all = {k: np.zeros((L, k, self.samples_per_k)) for k in range(2, self.neighbor_subset_limit + 1)}
    # where_k_neighbors = {k: [] for k in range(2, self.neighbor_subset_limit + 1)}
    # for l in range(L):
    #   neighbors_l = np.append(adjacency_lst[l], [l])
    #   N_l = np.min((len(neighbors_l), self.neighbor_subset_limit))
    #   for k in range(2, N_l + 1):
    #     permutations_k = list(permutations(neighbors_l, int(k)))
    #     where_k_neighbors[k].append(l)
    #     if self.samples_per_k is not None:
    #       permutations_k_ixs = np.random.choice(len(permutations_k), size=self.samples_per_k, replace=False)
    #       permutations_k = [permutations_k[ix] for ix in permutations_k_ixs]
    #     permutations_all[k][l, :] = np.array(permutations_k).T

    def fk(b, k):
      if k == 1:
        return self.h(b)
      else:
        result = torch.zeros((L, self.J))
        permutations_k = self.permutations_all[k]
        where_k_neighbors_ = self.where_k_neighbors[k]
        for perm_ix in range(self.samples_per_k):
          permutations_k_perm_ix = permutations_k[:, :, perm_ix]
          # ToDo: indices in where_k_neighbors_ will be wrong for k < neighbor_subset_limit, because X_ shrinks
          # X_1 = torch.tensor(X_[permutations_k_perm_ix[where_k_neighbors_, 0]])
          X_1 = X_[permutations_k_perm_ix[where_k_neighbors_, 0]]
          X_lst = np.column_stack([X_[permutations_k_perm_ix[where_k_neighbors_, ix]] for ix in range(1, k)])
          X_lst = torch.tensor(X_lst)
          fkm1_val = fk(X_lst, k-1)
          h_val = self.h(X_1)
          h_val_cat_fkm1_val = torch.cat((h_val, fkm1_val), dim=1)
          # ToDo: uncomment to learn binary relation instead of fixing to addition
          # g_val = self.g(h_val_cat_fkm1_val)
          g_val = h_val + fkm1_val
          result += g_val / len(permutations_k)
        return result

    E = fk(X_, self.neighbor_subset_limit)
    E = F.relu(E)
    E =  torch.cat((X_, E), dim=1)
    if locations_subset is not None:
      E = E[locations_subset]
    return E

  def forward_recursive(self, X_, adjacency_lst):
    L = X_.shape[0]
    final_ = torch.tensor([])
    X_ = torch.tensor(X_)
    # ToDo: vectorize
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
            # # ToDo: using fixed binary relation to see how it affects speed
            # g_val = h_val + fkm1_val
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


def evaluate_model_on_dataset(model, X_, y_, adjacency_list, sample_size, location_subsets=None,
                              targets_are_probs=False):
  mean_acc = 0.

  for ix, (X, y) in enumerate(zip(X_, y_)):
    if location_subsets is not None:
      location_subset = location_subsets[ix]
      output = model(torch.FloatTensor(X), adjacency_list, location_subset)
      if targets_are_probs:
        yhat = F.softmax(output)[:, 1].detach().numpy()
        acc = -((yhat - y[location_subset])**2).mean()
      else:
        yhat = F.softmax(output)[:, 1].detach().numpy()
        acc = ((yhat > 0.5) == y[location_subset]).mean()
    else:
      output = model(torch.FloatTensor(X), adjacency_list)
      if targets_are_probs:
        yhat = F.softmax(output)[:, 1].detach().numpy()
        acc = -((yhat - y)**2).mean()
      else:
        yhat = F.softmax(output)[:, 1].detach().numpy()
        acc = ((yhat > 0.5) == y).mean()
    mean_acc += acc / sample_size
  return mean_acc


def get_ggcn_val_objective(T, train_num, X_list, y_list, adjacency_list, n_epoch, nhid, batch_size, verbose,
                           neighbor_subset_limit, samples_per_k, recursive, target_are_probs=False):
  """
  Helper for tune_gccn. 
  """
  L = X_list[0].shape[0]

  def CV_objective(settings):
    lr = settings['lr']
    dropout = settings['dropout']

    # Split into train, val
    train_ixs = [np.random.choice(L, size=train_num, replace=False) for t in range(T)]
    val_ixs = [[l for l in range(L) if l not in train_ixs[t]] for t in range(T)]

    # Fit model on training data
    model = fit_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size, verbose=verbose,
             neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k, recursive=recursive, lr=lr,
                     dropout=dropout, locations_subsets=train_ixs, target_are_probs=target_are_probs)

    # Evaluate model on evaluation data
    sample_size = T
    total_val_acc = evaluate_model_on_dataset(model, X_list, y_list, adjacency_list, sample_size,
                                              location_subsets=val_ixs, targets_are_probs=target_are_probs)
    return total_val_acc, model

  return CV_objective


def learn_ggcn(X_list, y_list, adjacency_list, n_epoch=100, nhid=100, batch_size=5, verbose=False,
               neighbor_subset_limit=2, samples_per_k=6, recursive=True, num_settings_to_try=5,
               target_are_probs=False, lr=0.01, tol=0.01, dropout=0.2):

  if len(X_list) > 1:
    # _, model, _ = tune_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size,
    #                         verbose=verbose, neighbor_subset_limit=neighbor_subset_limit,
    #                         samples_per_k=samples_per_k, recursive=recursive, num_settings_to_try=num_settings_to_try,
    #                         X_holdout=None, y_holdout=None, target_are_probs=target_are_probs)
    model = fit_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size,
                     verbose=verbose, neighbor_subset_limit=neighbor_subset_limit,
                     samples_per_k=samples_per_k, recursive=recursive, lr=lr, tol=tol, dropout=dropout,
                     target_are_probs=target_are_probs)
  else:
    model = fit_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size,
                     verbose=verbose,neighbor_subset_limit=neighbor_subset_limit,
                     samples_per_k=samples_per_k, recursive=recursive, lr=0.01, tol=0.01, dropout=0.2,
                     target_are_probs=target_are_probs)

  def embedding_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    E = model.embed_recursive_vec(X_).detach().numpy()
    return E

  def model_wrapper(X_):
    X_ = torch.FloatTensor(X_)
    yhat = np.zeros(X_.shape[0])
    for _ in range(5):
      logits = model.forward_recursive_vec(X_, train=False)
      yhat_sample = F.softmax(logits, dim=1)[:, 1].detach().numpy()
      yhat += yhat_sample / 5
    return yhat

  return embedding_wrapper, model_wrapper


def ggcn_multiple_runs(X_raw_list, y_list, adjacency_list, env, eval_actions, true_probs, num_runs=5):
  best_model = None
  best_score = float('inf')
  best_phat = None
  for _ in range(num_runs):
    _, predictor = learn_ggcn(X_raw_list, y_list, adjacency_list)
    def qfn(a):
      return predictor(env.data_block_at_action(-1, a, raw=True))
    phat = np.hstack([qfn(a_) for a_ in eval_actions])
    # score = np.mean((phat - true_probs)**2)
    onem_phat = 1 - phat
    onem_true_probs = 1 - true_probs
    score = kl(phat, true_probs)
    if score < best_score:
      best_model = predictor
      best_score = score
      best_phat = phat
  return best_model, best_score, best_phat


def oracle_tune_ggcn(X_list, y_list, adjacency_list, env, eval_actions, true_probs,
                     X_eval=None,
                     n_epoch=70, nhid=100, batch_size=5, verbose=False,
                     samples_per_k=6, recursive=True, num_settings_to_try=3,
                     X_holdout=None, y_holdout=None, target_are_probs=False):
  """
  Tune GGCN hyperparameters, given sample of true probabilities evaluated at the current state.
  """
  # LR_RANGE = np.logspace(-3, -1, 100)
  # DROPOUT_RANGE = np.linspace(0, 1.0, 100)
  # NHID_RANGE = np.linspace(5, 30, 3)
  NEIGHBOR_SUBSET_LIMIT_RANGE = [2]
  
  LR_RANGE = [0.005]
  DROPOUT_RANGE = [0.0]
  NHID_RANGE = [16]
  

  best_predictor = None
  best_score = float('inf')
  worst_score = -float('inf')
  results = {'lr': [], 'dropout': [], 'nhid': [], 'neighbor_subset': [], 'score': []}
  for _ in range(num_settings_to_try):
    # Fit model with settings
    lr = np.random.choice(LR_RANGE)
    dropout = np.random.choice(DROPOUT_RANGE)
    nhid = int(np.random.choice(NHID_RANGE))
    neighbor_subset_limit = np.random.choice(NEIGHBOR_SUBSET_LIMIT_RANGE)

    print(f'lr: {lr} dropout: {dropout} nhid: {nhid} neighbor_limit: {neighbor_subset_limit}')

    _, predictor = learn_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=5, verbose=verbose,
                              neighbor_subset_limit=neighbor_subset_limit, samples_per_k=6, recursive=True, num_settings_to_try=5,
                              target_are_probs=False, lr=lr, tol=0.01, dropout=dropout)

    # Compare to true probs
    def qfn(a):
      # X_raw_ = env.data_block_at_action(-1, a, raw=True)
      X_ = env.data_block_at_action(-1, a)
      # if hasattr(env, 'NEIGHBOR_DISTANCE_MATRIX'):
      #   X_raw_ = np.column_stack((X_raw_, env.NEIGHBOR_DISTANCE_MATRIX))
      # else:
      #   X_raw_ = copy.copy(X_eval)
      #   X_raw_[:, 1] = a
      # return predictor(X_raw_)
      return predictor(X_)

    phat = np.hstack([qfn(a_) for a_ in eval_actions])
    score = kl(phat, true_probs)

    if score < best_score:
      best_score = score
      best_predictor = predictor
    if score > worst_score:
      worst_score = score
    print(f'best score: {best_score} worst score: {worst_score}')
    results['lr'].append(lr)
    results['dropout'].append(dropout)
    results['nhid'].append(nhid)
    results['neighbor_subset'].append(neighbor_subset_limit)
    results['score'].append(score)

  return best_predictor, results


def tune_ggcn(X_list, y_list, adjacency_list, n_epoch=50, nhid=100, batch_size=5, verbose=False,
              neighbor_subset_limit=2, samples_per_k=6, recursive=True, num_settings_to_try=5,
              X_holdout=None, y_holdout=None, target_are_probs=False):
  """
  Tune hyperparameters of GGCN; search over
    lr
    dropout
  """
  VAL_PCT = 0.3
  T = len(X_list)
  L = X_list[0].shape[0]
  TRAIN_NUM = int((1 - VAL_PCT) * L)
  LR_RANGE = np.logspace(-3, -1, 100)
  DROPOUT_RANGE = np.linspace(0, 0.5, 100)

  # Note that in training, splitting takes place over locations, whereas in holdout evaluation, splitting
  # takes place over timesteps
  if X_holdout is not None:
    holdout_size = len(X_holdout)

  objective = get_ggcn_val_objective(T, TRAIN_NUM, X_list, y_list, adjacency_list, n_epoch, nhid, batch_size, verbose,
                                     neighbor_subset_limit, samples_per_k, recursive, target_are_probs=target_are_probs)

  best_settings = None
  best_model = None
  best_value = -float('inf')
  worst_value = float('inf')
  all_models = []
  results = {i: {} for i in range(num_settings_to_try)}
  for i in range(num_settings_to_try):
    lr = np.random.choice(LR_RANGE)
    dropout = np.random.choice(DROPOUT_RANGE)
    settings = {'lr': lr, 'dropout': dropout}
    value, model = objective(settings)
    # print('settings: {} value: {}'.format(settings, value))
    if value > best_value:
      best_value = value
      best_settings = settings
      best_model = model
    if value < worst_value:
      worst_value = value

    results[i]['val'] = value

    if X_holdout is not None:
      holdout_acc = evaluate_model_on_dataset(model, X_holdout, y_holdout, adjacency_list, holdout_size)
      results[i]['holdout'] = holdout_acc

  if target_are_probs:
    baseline = -np.mean([np.var(y_) for y_ in y_list])
  else:
    baseline = np.mean([np.mean(y_) for y_ in y_list])
    baseline = np.max((baseline, 1-baseline))
  print(f'best value: {best_value} worst value: {worst_value} baseline: {baseline}')

  return best_settings, best_model, results


# ToDo: change name
# def learn_gccn(X_list, y_list, adjacency_list, n_epoch=10, nhid=100, batch_size=5, verbose=False,
#              neighbor_subset_limit=2, samples_per_k=6, recursive=True):
#
#   model = fit_ggcn(X_list, y_list, adjacency_list, n_epoch=n_epoch, nhid=nhid, batch_size=batch_size, verbose=verbose,
#                    neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k, recursive=recursive)
#
#   def embedding_wrapper(X_):
#     X_ = torch.FloatTensor(X_)
#     E = model.embed_recursive_vec(X_, adjacency_list).detach().numpy()
#     return E
#
#   def model_wrapper(X_):
#     X_ = torch.FloatTensor(X_)
#     logits = model.forward_recursive_vec(X_, adjacency_list)
#     yhat = F.softmax(logits)[:, 1].detach().numpy()
#     return yhat
#
#   return embedding_wrapper, model_wrapper


def fit_ggcn(X_list, y_list, adjacency_list, n_epoch=50, nhid=100, batch_size=5, verbose=True,
             neighbor_subset_limit=2, samples_per_k=6, recursive=True, lr=0.01, tol=0.001, dropout=0.0,
             locations_subsets=None, target_are_probs=False):
  # See here: https://github.com/tkipf/pygcn/blob/master/pygcn/train.py
  # Specify model
  p = X_list[0].shape[1]
  T = len(X_list)
  X_list = [torch.FloatTensor(X) for X in X_list]
  y_list = [torch.FloatTensor(y) for y in y_list]
  # model = GGCN(nfeat=p, J=nhid, neighbor_subset_limit=neighbor_subset_limit, samples_per_k=samples_per_k,
  #              recursive=recursive)
  model = GGCN(nfeat=p, J=nhid, adjacency_lst=adjacency_list, neighbor_subset_limit=neighbor_subset_limit,
               samples_per_k=samples_per_k,
               recursive=recursive, dropout=dropout, apply_sigmoid=target_are_probs)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
  if target_are_probs:
    criterion = nn.MSELoss()
  else:
    criterion = nn.CrossEntropyLoss()

  # Train
  for epoch in range(n_epoch):
    avg_acc_train = 0.
    batch_ixs = np.random.choice(T, size=batch_size)

    X_batch = [X_list[ix] for ix in batch_ixs]
    y_batch = [y_list[ix] for ix in batch_ixs]

    for X, y, ix in zip(X_batch, y_batch, batch_ixs):
      model.train()
      optimizer.zero_grad()
      X = Variable(X)
      if target_are_probs:
        y = Variable(y)
      else:
        y = Variable(y).long()

      if locations_subsets is not None:
        locations_subset = locations_subsets[ix]
        output = model(X, adjacency_list, locations_subset)
        if target_are_probs:
          output = output[:, 1]
        loss_train = criterion(output, y[locations_subset])
      else:
        output = model(X, adjacency_list)
        if target_are_probs:
          output = output[:, 1]
        loss_train = criterion(output, y)

      loss_train.backward()
      optimizer.step()

    # Evaluate loss
    for X_, y_ in zip(X_list, y_list):
      yhat = F.softmax(model(X_, adjacency_list), dim=1)[:, 1]
      if target_are_probs:
        acc = ((yhat - y_)**2).float().mean().detach().numpy()
      else:
        acc = ((yhat > 0.5) == y_).float().mean().detach().numpy()
      avg_acc_train += acc / T

    # Tracking diagnostics
    grad_norm = np.sqrt(np.sum([param.grad.data.norm(2).item()**2 for param in model.parameters() if param.grad
                                is not None]))
    yhat0 = model(X_list[0], adjacency_list)

    if verbose:
      print('Epoch: {:04d}'.format(epoch+1),
            'acc_train: {:.4f}'.format(avg_acc_train),
            'grad_norm: {:.4f}'.format(grad_norm))

    # Break if change in accuracy is sufficiently small
    # if epoch > 0:
    #   relative_acc_diff = np.abs(prev_avg_acc_train - avg_acc_train) / avg_acc_train
    #   if relative_acc_diff < tol:
    #     break

    prev_avg_acc_train = avg_acc_train

  if verbose:
    final_acc_train = 0.
    for X, y in zip(X_list, y_list):
      output = model(X, adjacency_list)
      yhat = F.softmax(output)[:, 1]
      acc = ((yhat > 0.5) == y).float().mean()
      final_acc_train += acc / T
    print('final_acc_train: {:.4f}'.format(final_acc_train))


  return model


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
  model = GCN(nfeat=p, nhid=nhid, nclass=2, dropout=0.8)
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
    # y1 = np.exp(y_[:, 1])
    return y_

  if verbose:
    final_acc_train = 0.
    for X, y in zip(X_list, y_list):
      output = model_wrapper(X)
      acc = ((output > 0.5) == y.detach().numpy()).mean()
      final_acc_train += acc / T
    print('final_acc_train: {:.4f}'.format(final_acc_train))

  return embedding_wrapper, model_wrapper


if __name__ == "__main__":
  # Test
  grid_size = 100
  adjacency_mat = random_nearest_neighbor(grid_size)
  adjacency_list = [[j for j in range(grid_size) if adjacency_mat[i, j]] for i in range(grid_size)]
  neighbor_counts = adjacency_mat.sum(axis=1)
  n = 10
  n_epoch = 200
  X_list = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(n)])
  y_probs_list = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in X_list])
  y_list = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in y_probs_list])

  # Holdout data
  X_list_holdout = np.array([np.random.normal(size=(grid_size, 2)) for _ in range(2)])
  y_probs_list_holdout = np.array([np.array([expit(np.sum(X[adjacency_list[l]])) for l in range(grid_size)]) for X in
                                   X_list_holdout])
  y_list_holdout = np.array([np.array([np.random.binomial(1, prob) for prob in y_probs]) for y_probs in
                             y_probs_list_holdout])


  print('Fitting ggcn')
  _, model_ = learn_ggcn(X_list, y_list, adjacency_list, n_epoch=50, nhid=100, batch_size=5, verbose=False,
             neighbor_subset_limit=2, samples_per_k=6, recursive=True, num_settings_to_try=5,
             target_are_probs=False)

  oracle_mean = 0.
  ggcn_mean = 0.
  baseline_mean = 0.
  for X, yp, y in zip(X_list_holdout, y_probs_list_holdout, y_list_holdout):
    y_hat_oracle = (yp > 0.5)
    y_hat_ggcn = (model_(X) > 0.5)
    y_hat_baseline = (np.mean(y) > 0.5)
    oracle_mean += (y_hat_oracle == y).mean() / len(y_list_holdout)
    ggcn_mean += (y_hat_ggcn == y).mean() / len(y_list_holdout)
    baseline_mean += (y_hat_baseline == y).mean() / len(y_list_holdout)
  print(f'oracle: {oracle_mean} ggcn: {ggcn_mean} baseline: {baseline_mean}')
