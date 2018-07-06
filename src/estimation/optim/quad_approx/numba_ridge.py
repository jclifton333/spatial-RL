from numba import jit
import numpy as np


@jit(nopython=True)
def inv(A):
  return np.linalg.inv(A)

@jit(nopython=True)
def mat_vec(A, v):
  m, n = A.shape
  new_vec = np.zeros(m)
  for i in range(m):
    for j in range(n):
      new_vec[i] += A[i,j]*v[j]
  return new_vec


@jit(nopython=True)
def mat_mat(A, B):
  m, n = A.shape
  p = B.shape[0]
  C = np.zeros((m, p))


