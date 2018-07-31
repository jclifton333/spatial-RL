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
from src.environments.sis_infection_probs import infection_probability
from src.environments import generate_network, SIS
from src.estimation.model_based.SIS.simulate import simulate_from_SIS
from src.utils.misc import KerasLogit, SKLogit
from functools import partial
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


def loss(r0, r1, mean_infection_prop):
  """
  :param r0:
  :param r1: estimated mse ratio when eps = 1; we want this to be close to 2, i.e. MF is twice as accurate under
             full contamination.
  :param mean_infection_prop:
  :return:
  """
  # epsilon_list = (0.25, 0.5, 0.75, 1)
  # closeness_loss = np.mean([(r(epsilon, r0) - r_epsilon)**2 for epsilon, r_epsilon in
  #                           zip(epsilon_list, r_epsilon_list)])
  # order_loss = np.mean(np.array(r_epsilon_list[1:]) - np.array(r_epsilon_list[:-1]) > 0)
  closeness_loss = (r1 - 2)**2
  infection_rate_loss = (mean_infection_prop < 0.3) + (mean_infection_prop > 0.5)
  return closeness_loss + infection_rate_loss


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
                                    time_horizon=25, n_rep=10):
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

  a = np.concatenate((np.zeros(47), np.ones(3)))
  mb_loss = mf_loss = 0
  for rep in range(n_rep):
    env.reset()
    env.step(np.random.permutation(a))
    env.step(np.random.permutation(a))
    mb_loss_rep = mf_loss_rep = 0
    for t in range(time_horizon):
      env.step(np.random.permutation(a))

      # Fit models
      features = np.vstack(env.X)
      target = np.hstack(env.y).astype(float)
      eta = fit_transition_model(env)
      mf_model = mf_constructor()
      mf_model.fit(features, target, weights=None)

      # Compute losses
      true_probs = np.hstack(env.true_infection_probs)
      phat_mb = np.zeros(0)
      for x, t in zip(env.X_raw, range(len(env.X_raw))):
        s, a, y = x[:, 0], x[:, 1], x[:, 2]
        phat_mb_t = env.infection_probability(a, s, y, eta=eta)
        phat_mb = np.append(phat_mb, phat_mb_t)
      phat_mf = mf_model.predict_proba(features)[:, -1]
      mb_loss_t = np.sum((phat_mb - true_probs)**2)
      mf_loss_t = np.sum((phat_mf - true_probs)**2)
      mb_loss_rep += (mb_loss_t - mb_loss_rep) / (t + 1)
      mf_loss_rep += (mf_loss_t - mf_loss_rep) / (t + 1)
      K.clear_session()
    mb_loss += (mb_loss_rep - mb_loss) / (rep + 1)
    mf_loss += (mf_loss_rep - mf_loss) / (rep + 1)
  return mb_loss / mf_loss


def simulate_data_to_compare_on(contamination_weight_vector, contaminator_constructor, time_horizon=25, n_rep=10):
  """
  Simulate from model epsilon=1 with myopic policy and return observations; this will be to compare MSEs of fitted
  models.
  :param epsilon:
  :param contamination_weight_vector: [coef, bias]
  :return:
  """
  contaminator = contaminator_constructor()
  coef, bias = contamination_weight_vector[:-1], contamination_weight_vector[-1]
  contaminator.set_weights([coef.reshape(-1, 1), np.array(bias).reshape(1)], len(coef))

  sis_kwargs = {'L': 50, 'omega': 0, 'generate_network': generate_network.lattice,
                'initial_infections': None, 'add_neighbor_sums': False, 'contaminator': contaminator,
                'epsilon': 1}

  # env = SIS.SIS(50, 0, generate_network.lattice, )
  env = environment_factory('SIS', **sis_kwargs)
  simulation_env = simulate_from_SIS(env, env.eta, time_horizon, 3)
  return simulation_env


def simulate_and_get_loss(contamination_weight_vector, r0, contaminator_constructor, mf_constructor, time_horizon=10,
                          n_rep=25):
  reference_env = simulate_data_to_compare_on(contamination_weight_vector, contaminator_constructor,
                                              time_horizon=time_horizon, n_rep=1)
  true_probs = np.hstack(reference_env.true_infection_probs)
  reference_features = np.vstack(reference_env.X)
  phat_mb_array = np.zeros((0, len(true_probs)))
  phat_mf_array = np.zeros((0, len(true_probs)))
  for rep in range(n_rep):
    print(rep)
    simulation_env = simulate_data_to_compare_on(contamination_weight_vector, contaminator_constructor,
                                                 time_horizon=time_horizon, n_rep=1)
    # Fit models
    features = np.vstack(simulation_env.X)
    target = np.hstack(simulation_env.y).astype(float)
    eta = fit_transition_model(simulation_env)
    mf_model = mf_constructor()
    mf_model.fit(features, target, weights=None)

    # Compute predicted probs
    phat_mf = mf_model.predict_proba(reference_features)[:, -1]
    phat_mb = np.zeros(0)
    for t, x in enumerate(reference_env.X_raw):
      s, a, y = x[:, 0], x[:, 1], x[:, 2]
      phat_mb_t = infection_probability(a, s, y, eta, 0, reference_env.L, reference_env.adjacency_list)
      phat_mb = np.append(phat_mb, phat_mb_t)
    K.clear_session()

    # Add to array
    phat_mb_array = np.vstack((phat_mb_array, phat_mb))
    phat_mf_array = np.vstack((phat_mf_array, phat_mf))

  # Compute bias and variance
  mb_bias = np.mean(np.mean(phat_mb_array, axis=0) - true_probs)
  mb_variance = np.mean(np.var(phat_mb_array, axis=0))
  mf_bias = np.mean(np.mean(phat_mf_array, axis=0) - true_probs)
  mf_variance = np.mean(np.var(phat_mf_array, axis=0))
  print('mb bias: {} mb var: {} mf bias: {} mf var: {}'.format(mb_bias, mb_variance, mf_bias, mf_variance))
  r1 = (mb_bias**2 + mb_variance) / (mf_bias**2 + mf_variance)
  mean_infection_prop = np.mean(reference_env.y)
  loss_ = loss(r0, r1, mean_infection_prop)
  return loss_, mean_infection_prop, r1


def do_initial_sampling_and_get_losses(r0, initial_weight_list=None, sample_weights_and_epsilons=None,
                                       mf_constructor=SKLogit, contaminator_constructor=KerasLogit, n_samples=50):
  if initial_weight_list is None and sample_weights_and_epsilons is None:
    initial_weight_list = fit_mf_estimator_to_uncontaminated_sis(mf_constructor_name=mf_constructor)
  if sample_weights_and_epsilons is None:
    sample_weights_and_epsilons = sample_weights(initial_weight_list, n_samples=n_samples)
  losses = []
  r_epsilon_lists = []
  res = {}
  sample_counter = 0
  for _, contamination_weight_vector in sample_weights_and_epsilons:
    r_epsilon_list = []
    for epsilon in (0.25, 0.5, 0.75, 1):
      r_eps = simulate_and_estimate_mse_ratio(epsilon, contamination_weight_vector, mf_constructor=mf_constructor,
                                              contaminator_constructor=contaminator_constructor)
      r_epsilon_list.append(r_eps)
    loss_ = loss(r0, r_epsilon_list)
    losses.append(loss_)
    r_epsilon_lists.append(r_epsilon_list)
    res[sample_counter] = {'contamination_model_parameter': contamination_weight_vector,
                           'r_epsilon_list': r_epsilon_list, 'loss': loss_}
    sample_counter += 1
  save_in_tuning_data(res, 'initial-sample-ratios-and-losses', 'pkl')
  return sample_weights_and_epsilons, r_epsilon_lists, losses


if __name__ == '__main__':
  np.random.seed(SEED)
  # r0 = get_r0()
  r0 = 0.01
  # sample_weights_and_epsilons_fname = \
  #   os.path.join(this_dir, 'tuning_data', 'sis-tuning-initial-sample-weights-180728_184731.p')
  # sample_weights_and_epsilons = pkl.load(open(sample_weights_and_epsilons_fname, 'rb'))['samples']
  # ans = do_initial_sampling_and_get_losses(r0, sample_weights_and_epsilons=sample_weights_and_epsilons)
  # print(ans[1], ans[2])
  loss_function = partial(simulate_and_get_loss, r0=r0, contaminator_constructor=KerasLogit,
                          mf_constructor=SKLogit, n_rep=5)
  counter = 0
  max_counter = 10
  mean_inf = 0
  intercept_ = np.array([-2.2])
  while counter < max_counter and (mean_inf > 0.5 or mean_inf < 0.3):
    coef_ = np.random.normal(size=16, loc=0.1, scale=0.5)
    print(coef_)
    weight = np.concatenate((coef_, intercept_))
    res = loss_function(weight)
    print(res)
    counter += 1
    mean_inf = res[1]
  best_coef_ = coef_
  best_res = res
  while counter < max_counter:
    new_coef_ = np.random.normal(loc=best_coef_)
    weight = np.concatenate((new_coef_, intercept_))
    print(res)
    res = loss_function(weight)
    if res[0] < best_res[0]:
      best_res = res
      best_coef = new_coef_
    counter += 1
  print('best res {}'.format(best_res), 'best coef {}'.format(best_coef_))




