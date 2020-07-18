# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..')
sys.path.append(pkg_dir)

from src.estimation.q_functions.embedding import learn_gcn
import numpy as np
from sklearn.linear_model import LogisticRegression
import pdb

if __name__ == "__main__":
  fname = os.path.join('observations', os.listdir('./observations')[0])
  data = np.load(fname, allow_pickle=True)
  X_raw_list = data[()]['X_raw']
  y_list = data[()]['y']
  adjacency_mat = data[()]['adjacency_mat']

  # Fit gcn
  embedder, _ = learn_gcn(X_raw_list, y_list, adjacency_mat, nhid=50, verbose=True, n_epoch=50)

  # Naive baseline
  clf = LogisticRegression()
  X_raw = np.vstack(X_raw_list)
  y = np.hstack(y_list)
  clf.fit(X_raw, y)
  print('Naive logit: {}'.format(clf.score(X_raw, y)))

  # Logit with embeddings
  X_embedded = np.vstack([embedder(x_raw) for x_raw in X_raw_list])
  clf1 = LogisticRegression()
  clf.fit(X_embedded, y)
  print('Embedded logit: {}'.format(clf.score(X_embedded, y)))


