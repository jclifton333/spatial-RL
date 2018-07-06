"""
Gradients for (log) likelihood of SIS generative model.
State transitions are parameterized by beta = [beta_0, beta_1].
Infection transitions are parameterized by eta = [eta_0, ..., eta_6].
"""
import numpy as np


def exp_logit_q_l(a, eta):
  return eta[5] + eta[6]*a


def q_l_grad(a, exp_logit_q_l_):
  multiplier = 1 - exp_logit_q_l_ / (1 + exp_logit_q_l_)
  return np.multiply(multiplier, a)



