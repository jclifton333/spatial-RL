# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..')
sys.path.append(pkg_dir)

import src.estimation.q_functions.embedding as embed
import numpy as np
from sklearn.linear_model import LogisticRegression
import pdb

if __name__ == "__main__":
  fit_gcn = False
  fit_naive = False
  fit_ggcn = True

  fname = os.path.join('observations', os.listdir('./observations')[0])
  data = np.load(fname, allow_pickle=True)
  X_raw_list = data[()]['X_raw']
  y_list = data[()]['y']
  y = np.hstack(y_list)
  adjacency_mat = data[()]['adjacency_mat']
  L = adjacency_mat.shape[0]
  adjacency_list = [[lprime for lprime in range(L) if adjacency_mat[l, lprime]] for l in range(L)]

  if fit_gcn:
    # Fit gcn
    embedder, _ = embed.learn_gcn(X_raw_list, y_list, adjacency_mat, nhid=10, verbose=True, n_epoch=20)

    # Logit with embeddings
    X_embedded = np.vstack([embedder(x_raw) for x_raw in X_raw_list])
    clf1 = LogisticRegression()
    clf1.fit(X_embedded, y)
    print('Embedded logit: {}'.format(clf1.score(X_embedded, y)))

  if fit_naive:
    # Naive baseline
    clf = LogisticRegression()
    X_raw = np.vstack(X_raw_list)
    y = np.hstack(y_list)
    clf.fit(X_raw, y)
    print('Naive logit: {}'.format(clf.score(X_raw, y)))

    # Logit with naive embeddings
    h = lambda x_: x_
    g = lambda x_, y_: x_ + y_
    X_naive_embedded = np.vstack([embed.embed_network(x_raw, adjacency_list, g, h, 3) for x_raw in X_raw_list])
    clf2 = LogisticRegression()
    clf2.fit(X_naive_embedded, y)
    print('Naive embedded logit: {}'.format(clf2.score(X_naive_embedded, y)))

  if fit_ggcn:
    nhid = 20
    # embed.learn_ggcn(X_raw_list, y_list, adjacency_list, n_epoch=100, verbose=True, batch_size=10,
    #                  neighbor_subset_limit=1, nhid=nhid)
    embed.learn_ggcn(X_raw_list, y_list, adjacency_list, n_epoch=100, verbose=True, batch_size=10,
                     neighbor_subset_limit=3, nhid=nhid, samples_per_k=2)

