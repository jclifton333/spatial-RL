"""
To see the effects of model misspecification on MB vs. MF approaches, we examine models of the form
\epsilon * SIS_transition_probs + (1 - \epsilon) * contamination_transition_probs.

contamination_transition_probs will be output by a neural network, which is tuned as follows:

1. Estimate r(0) = MSE( \hat{p}_MB | \epsilon=0) / MSE( \hat{p}_MF | \epsilon=0 ), i.e. the relative performance of
   MB and MF one-step probability estimates when the SIS model is uncontaminated, for a size 50 lattice integrated
   over 25 time steps.
2. For \epsilon = 0.25, 0.75, 1, randomly generate contamination network parameters, and estimate relative MSEs
   for each.
3. Use Gaussian process optimization (https://thuijskens.github.io/2016/12/29/bayesian-optimisation/)
   to find contamination parameter that minimizes \lVert r(\epsilon; \beta) - r(\epsilon) \rVert^2,
   where r(\epsilon; \beta) is the observed ratio of MSEs and r(\epsilon) is the desired ratio - a line
   with slope (1/r(0) - r(0)).
"""
# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..')
sys.path.append(pkg_dir)
import numpy as np
import yaml
from src.environments.environment_factory import environment_factory
from src.environments import generate_network
from src.estimation.model_based.SIS.simulate import simulate_from_SIS

# yml containing MSE info for uncontaminated SIS model
SAVED_MSE_FNAME = '../../analysis/results/prob_estimates_SIS_random_quad_approx_50_0.0_180728_125347.yml'


def get_r0():
  """
  Get estimated ratio of MB to MF probability loss, under no-contamination.
  :return:
  """
  d = yaml.load(open(SAVED_MSE_FNAME))
  r0 = 0
  for rep, rep_results in d['results'].items():
    mb_loss = np.mean(rep_results['mb_loss'])
    mf_loss = np.mean(rep_results['obs_data_loss_KerasLogit'])
    r0_rep = mb_loss / mf_loss
    r0 += (r0_rep - r0) / (rep + 1)
  return r0


def r(epsilon, r0):
  """
  Get desired MSE(MB) / MSE(MF) under epsilon-contamination, where r0 is no-contamination.
  :param epsilon:
  :param r0:
  :return:
  """
  return (1 / r0 - r0) * epsilon + r0


def fit_mf_estimator_to_uncontaminated_sis(mf_constructor_name, time_horizon=25, n_rep=5):
  """
  Simulate from SIS omega=1, fit MF model, and save weights; these will be used to initialize search for
  contamination model weights.
  :return:
  """
  # Set up simulator and run
  sis_kwargs = {'L': 100, 'omega': 1, 'generate_network': generate_network.lattice,
                'initial_infections': None, 'add_neighbor_sums': False}
  env = environment_factory('SIS', **sis_kwargs)
  simulation_env = simulate_from_SIS(env, env.eta, time_horizon, None, None, 5, n_rep=n_rep)

  # Fit model
  sim_features = np.vstack(simulation_env.X)
  sim_target = np.hstack(simulation_env.y).astype(float)
  model = mf_constructor_name()
  model.fit(sim_features, sim_target, weights=None)

  # Save weights (note that "weights" is overloaded - sample (bootstrap) weights and coefficients)
  fitted_weights = [model.coef_, model.intercept_]
  fname = os.path.join(this_dir, 'tuning_data/uncontaminated-fitted-mf-weights.yml')
  with open(fname, 'w') as outfile:
    yaml.dump({'weights': fitted_weights}, outfile)
  return


def sample_weights(initial_weight_list, n_samples=50):
  """

  :param initial_weight_list: [mf_coef, mf_intercept] about which candidate contamination parameters will be sampled.
  :return:
  """


