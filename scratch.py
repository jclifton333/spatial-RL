import numpy as np
from scipy.special import expit
import pdb


def prob(eta0, eta1, eta2, eta3, eta4, s0, s1, d, a0, a1):
  gravity_num = np.exp(eta1) * d
  gravity_denom = np.power(s0 * s1, eta2)
  logit_prob = eta0 - (gravity_num / gravity_denom) + eta3*a0 + eta4*a1
  prob = expit(logit_prob)
  print('prob: {} gravity num: {} gravity denom: {}'.format(prob, gravity_num, gravity_denom))
  return prob


if __name__ == '__main__':
  eta0 = -6.5
  eta1 = np.log(156)
  eta2 = 1.0
  eta3 = -8
  eta4 = -8
  s0 = s1 = 10
  d = 10
  a0 = 0
  a1 = 0
  p = prob(eta0, eta1, eta2, eta3, eta4, s0, s1, d, a0, a1)
  print(1 - np.power(1 - p, 290))