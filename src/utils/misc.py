from functools import partial
import numpy as np


def onehot(length, ix):
  arr = np.zeros(length)
  arr[ix] = 1
  return arr


def kl(p, q):
  onem_p = 1 - p
  onem_q = 1 - q
  kl_ = (p * np.log(p / q) + onem_p * np.log(onem_p / onem_q)).mean()
  return kl_


def random_argsort(arr, num_to_take):
  """
  Ad-hoc way of getting randomized argsort.
  """
  top_entries = np.sort(-arr)[:(num_to_take*2)]
  b = np.random.random(top_entries.size)
  return np.argsort(np.lexsort((b, top_entries)))[:num_to_take]



