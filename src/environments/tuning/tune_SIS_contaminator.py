"""
To see the effects of model misspecification on MB vs. MF approaches, we examine models of the form
\epsilon * SIS_transition_probs + (1 - \epsilon) * contamination_transition_probs.

contamination_transition_probs will be output by a neural network, which is tuned as follows:

1. Estimate r(0) = MSE( \hat{p}_MB | \epsilon=0) / MSE( \hat{p}_MF | \epsilon=0 ), i.e. the relative performance of
   MB and MF one-step probability estimates when the SIS model is uncontaminated, for a size 50 lattice integrated
   over 25 time steps.
2. For \epsilon = 0.25, 0.75, 1, randomly generate contamination network parameters, and estimate relative MSEs
   for each.
3. Use ?
   to find contamination parameter that minimizes \lVert r(\epsilon; \beta) - r(\epsilon) \rVert^2,
   where r(\epsilon; \beta) is the observed ratio of MSEs and r(\epsilon) is the desired ratio - a line
   with slope (2 - r(0)) so that at full maximum contamination, MF MSE is half that of MB.
"""
import pdb
# Hack bc python imports are stupid
import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.join(this_dir, '..', '..', '..')
sys.path.append(pkg_dir)
import numpy as np
import datetime
import yaml
import pickle as pkl
from src.estimation.model_based.SIS.fit import fit_transition_model
from src.environments.environment_factory import environment_factory
from src.environments import generate_network, SIS
from src.estimation.model_based.SIS.simulate import simulate_from_SIS
from src.utils.misc import KerasLogit
import keras.backend as K

SEED = 3
# yml containing MSE info for uncontaminated SIS model
SAVED_MSE_FNAME = '../../analysis/results/prob_estimates_SIS_random_quad_approx_50_0.0_180728_125347.yml'


def get_r0():
  """
  Get estimated ratio of MB to MF probability loss, under no-contamination.
  :return:
  """
  d = yaml.load(open(SAVED_MSE_FNAME))
  mb_loss = mf_loss = 0
  for rep, rep_results in d['results'].items():
    mb_loss_rep = np.mean(rep_results['mb_loss'])
    mf_loss_rep = np.mean(rep_results['obs_data_loss_KerasLogit'])
    mb_loss += (mb_loss_rep - mb_loss) / (rep + 1)
    mf_loss += (mf_loss_rep - mf_loss) / (rep + 1)
  return mb_loss / mf_loss


def r(epsilon, r0):
  """
  Get desired MSE(MB) / MSE(MF) under epsilon-contamination, where r0 is no-contamination.
  :param epsilon:
  :param r0:
  :return:
  """
  return (2 - r0) * epsilon + r0


def save_in_tuning_data(dict_, basename, ftype):
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  if ftype == 'pkl':
    fname = 'tuning_data/{}-{}.{}'.format(basename, suffix, 'p')
  elif ftype == 'yml':
    fname = 'tuning_data/{}-{}.{}'.format(basename, suffix, 'yml')

  fname = os.path.join(this_dir, fname)
  if ftype == 'yml':
    with open(fname, 'w') as outfile:
      yaml.dump(dict_, outfile)
  elif ftype == 'pkl':
    pkl.dump(dict_, open(fname, 'wb'))
  return


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
  res = {'weights': fitted_weights}
  save_in_tuning_data(res, 'uncontaminated-fitted-mf-weights', 'pkl')
  return fitted_weights


def sample_weights(initial_weight_list, epsilon_list=(0.25, 0.5, 0.75, 1), n_samples=50):
  """

  :param initial_weight_list: [mf_coef, mf_intercept] about which candidate contamination parameters will be sampled.
  :param epsilon_list:
  :return: nsample-length list of tuples (\epsilon, sample coef)
  """
  initial_weight_vector = np.concatenate((initial_weight_list[0].flatten(), initial_weight_list[1]))
  samples = []
  for s in range(n_samples):
    eps = np.random.choice(epsilon_list)
    sample_weight_vector = np.random.normal(loc=initial_weight_vector, scale=5.0)
    samples.append((eps, sample_weight_vector))
  save_in_tuning_data({'samples': samples}, 'sis-tuning-initial-sample-weights', 'pkl')
  return samples


def simulate_and_estimate_mse_ratio(epsilon, contamination_weight_vector, mf_constructor, contaminator_constructor,
                                    time_horizon=25, n_rep=5):
  """
  Simulate from model epsilon * SIS + (1 - epsilon) * contaminated, fit MB and MF probs and return MSE ratio.
  :param epsilon:
  :param contamination_weight_vector: [coef, bias]
  :return:
  """
  contaminator = contaminator_constructor()
  coef, bias = contamination_weight_vector[:-1], contamination_weight_vector[-1]
  contaminator.set_weights([coef.reshape(-1, 1), np.array(bias).reshape(1)], len(coef))

# Set up simulator
  sis_kwargs = {'L': 50, 'omega': 0, 'generate_network': generate_network.lattice,
                'initial_infections': None, 'add_neighbor_sums': False, 'contaminator': contaminator,
                'epsilon': epsilon}

  # env = SIS.SIS(50, 0, generate_network.lattice, )
  env = environment_factory('SIS', **sis_kwargs)
  mf_model = mf_constructor()

  r_eps = 0
  a = np.concatenate((np.zeros(47), np.ones(3)))
  for rep in range(n_rep):
    env.reset()
    r_eps_rep = 0
    env.step(np.random.permutation(a))
    env.step(np.random.permutation(a))
    for t in range(time_horizon):
      # Fit models
      features = np.vstack(env.X)
      target = np.hstack(env.y).astype(float)
      eta = fit_transition_model(env)
      mf_model.fit(features, target, weights=None)

      # Compute losses
      true_probs = np.hstack(env.true_infection_probs)
      phat_mb = env.next_infection_probabilities(a, eta=eta)
      phat_mf = mf_model.predict_proba(env.X[-1])[:, -1]
      loss_mb = np.sum((phat_mb - true_probs)**2)
      loss_mf = np.sum((phat_mf - true_probs)**2)
      r_eps_rep += (loss_mb/loss_mf - r_eps_rep) / (t + 1)

      K.clear_session()
      env.step(np.random.permutation(a))
    r_eps += (r_eps_rep - r_eps) / (rep + 1)
  return r_eps


def do_initial_sampling_and_get_losses(r0, initial_weight_list=None, sample_weights_and_epsilons=None,
                                       mf_constructor=KerasLogit, contaminator_constructor=KerasLogit, n_samples=50):
  if initial_weight_list is None and sample_weights_and_epsilons is None:
    initial_weight_list = fit_mf_estimator_to_uncontaminated_sis(mf_constructor_name=mf_constructor)
  if sample_weights_and_epsilons is None:
    sample_weights_and_epsilons = sample_weights(initial_weight_list, n_samples=n_samples)
  losses = []
  for eps, contamination_weight_vector in sample_weights_and_epsilons:
    r_eps = simulate_and_estimate_mse_ratio(eps, contamination_weight_vector, mf_constructor=mf_constructor,
                                            contaminator_constructor=contaminator_constructor)
    r_desired = r(eps, r0)
    loss = (r_eps - r_desired)**2
    losses.append(loss)
  return sample_weights_and_epsilons, loss_ratios


if __name__ == '__main__':
  np.random.seed(SEED)
  # r0 = get_r0()
  r0 = 0.01
  sample_weights_and_epsilons_fname = \
    os.path.join(this_dir, 'tuning_data', 'sis-tuning-initial-sample-weights-180728_184731.p')
  sample_weights_and_epsilons = pkl.load(open(sample_weights_and_epsilons_fname, 'rb'))['samples']
  do_initial_sampling_and_get_losses(r0, sample_weights_and_epsilons=sample_weights_and_epsilons)




