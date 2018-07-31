import numpy as np
from scipy.special import expit


def random_contamination(X):
  L = X.shape[0]
  beta = np.random.random(L)
  L_beta = np.dot(L, beta)
  return expit(L_beta)
