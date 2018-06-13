"""
This is for testing the one-step lookahead regression.
"""


import numpy as np
import pdb

from src.environments.generate_network import lattice
from src.environments.Ebola import Ebola
from src.environments.SIS import SIS

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression

def main(L, T, nRep, envName, method='random'):
  """
  :param L: number of locations in network
  :param envName: 'SIS' or 'Ebola'
  :param T: duration of simulation rep
  :param nRep: number of replicates
  :param method: string in ['random', 'none']
  """
  # Initialize generative model
  omega = 0
  gamma = 0.7
  # featureFunction = polynomialFeatures(3, interaction_only=True)
  featureFunction = lambda d: d

  if envName == 'SIS':
    env = SIS(L, omega, featureFunction, lattice)
  elif envName == 'Ebola':
    env = Ebola(featureFunction)
  else:
    raise ValueError("Env name not in ['SIS', 'Ebola']")

  # Evaluation limit parameters
  treatment_budget = np.int(np.floor((3/16) * L))

  reg = AdaBoostRegressor(n_estimators=1000)

  a_dummy = np.append(np.ones(treatment_budget), np.zeros(env.L - treatment_budget))
  for rep in range(nRep):
    print('Rep: {}'.format(rep))
    env.reset()
    a = np.random.permutation(a_dummy)
    env.step(a)
    a = np.random.permutation(a_dummy)
    for i in range(T-2):
      env.step(a)
      if method == 'random':
        a = np.random.permutation(a_dummy)
      elif method == 'none':
        a = np.zeros(g.L)

      # One-step regression
      target = np.sum(env.y, axis=1).astype(float)
      true_expected_counts = np.sum(env.true_infection_probs, axis=1)
      reg.fit(np.array(env.Phi), target)
      phat = reg.predict(np.array(env.Phi))
      r2 = 1 - ( np.mean((phat - true_expected_counts)**2) / np.sum( (true_expected_counts - np.mean(true_expected_counts))**2) )
      print('R2: {}'.format(r2))
      # if r2 > 0.6:
      #   pdb.set_trace()

  return


if __name__ == '__main__':
  L = 16
  T = 100000
  nRep = 5
  envName = 'SIS'
  main(L, T, nRep, envName, method='random')
