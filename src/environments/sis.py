"""
Implementing susceptible-infected-susceptible (sis) models described in
spatial QL paper.
"""

import copy
import numpy as np
from src.estimation.model_based.sis.infection_model_objective import success_or_failure_component
from .SpatialDisease import SpatialDisease
from .sis_contaminator import SIS_Contaminator, recoding_mapping
from .sis_infection_probs import sis_infection_probability
from scipy.linalg import block_diag
import src.utils.gradient as gradient
import pdb

import os
this_dir = os.path.dirname(os.path.abspath(__file__))
tuning_data_dir = os.path.join(this_dir, 'tuning', 'tuning_data')


class SIS(SpatialDisease):
  ENCODING_MATRIX = np.array([np.power(2.0, 3-j) for j in range(1, 3+1)])

  # Fixed generative model parameters
  BETA_0 = 0.9
  BETA_1 = 1.0
  BETA = np.array([BETA_0, BETA_1])
  INITIAL_INFECT_PROB = 0.1

  """
  Parameters for infection probabilities.  See
    bin/run_infShieldState.cpp
    main/infShieldStatePosImNoSoModel.cpp
  
  Correspondence between draft and stmMF_cpp parameter names
    intcp_inf_latent_ : ETA_0
    trt_pre_inf_      : ETA_1, ETA_3
    intcp_inf_        : ETA_2
    trt_act_inf_      : ETA_4
    intcp_rec_        : ETA_5
    trt_act_rec_      : ETA_6
  """
  PROB_INF_LATENT = 0.01
  PROB_INF = 0.5
  PROB_NUM_NEIGH = 3
  PROB_REC = 0.25

  ETA_0 = np.log(1 / (1 - PROB_INF_LATENT) - 1)
  ETA_2 = np.log(((1 - PROB_INF) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1)
  ETA_4 = np.log(((1 - PROB_INF * 0.25) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1) - ETA_2
  ETA_3 = np.log(((1 - PROB_INF * 0.75) / (1 - PROB_INF_LATENT))**(-1 / PROB_NUM_NEIGH) \
          - 1) - ETA_2
  ETA_5 = np.log(1 / (1 - PROB_REC) - 1)
  ETA_6 = np.log(1 / ((1 - PROB_REC) * 0.5) - 1) - ETA_5
  ETA = np.array([ETA_0, ETA_3, ETA_2, ETA_3, ETA_4, ETA_5, ETA_6])

  # Contamination model stuff
  CONTAMINATOR = SIS_Contaminator()
  CONTAMINATION_MODEL_PARAMETER = np.array([
    -1.33, -1.44, -0.97, -0.56, 0.99, 0.70, 0.58, 1.38, -0.10, -0.19, -0.37, 0.02, 0.53, 0.64, 0.76, 1.06, -0.66
  ])
  CONTAMINATOR.set_weights(CONTAMINATION_MODEL_PARAMETER)

  def __init__(self, L, omega, generate_network, add_neighbor_sums=False, adjacency_matrix=None,
               initial_infections=None, initial_state=None, eta=None, beta=None,
               epsilon=0, contaminator=CONTAMINATOR):
    """
    :param omega: parameter in [0,1] for mixing two sis models
    :param generate_network: function that accepts network size L and returns adjacency matrix
    """
    self.ENCODING_DICT = {
          s: {
            a: {
              y: int(np.dot(np.array([s, a, y]), SIS.ENCODING_MATRIX)) for y in range(2)
            }
            for a in range(2)
          }
          for s in range(2)
    }
    self.add_neighbor_sums = add_neighbor_sums
    self.epsilon = epsilon
    self.contaminator = contaminator

    if eta is None:
      self.eta = SIS.ETA
    else:
      self.eta = eta

    if beta is None:
      self.beta = SIS.BETA
    else:
      self.beta = beta

    if adjacency_matrix is None:
      self.adjacency_matrix = generate_network(L)
    else:
      self.adjacency_matrix = adjacency_matrix

    self.lambda_ = self.adjacency_matrix
    SpatialDisease.__init__(self, self.adjacency_matrix, initial_infections)

    if initial_state is None:
      self.initial_state = np.zeros(self.L)
    else:
      self.initial_state = initial_state

    self.omega = omega
    self.state_covariance = self.beta[1] * np.eye(self.L)

    self.X_2 = []  # For second-order neighbor features
    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = []  # Network-level features
    self.current_state = self.S[-1, :]

    # For computing likelihood
    counts_for_likelihood_names = ['n_00_0', 'n_01_0', 'n_10_0', 'n_11_0', 'n_00_1', 'n_01_1', 'n_10_1', 'n_11_1',
                                   'a_0', 'a_1']
    self.counts_for_likelihood = {count_name: [] for count_name in counts_for_likelihood_names}

  def reset(self):
    """
    Reset state and observation histories.
    """
    super(SIS, self).reset()
    self.X_2 = []
    self.S = np.array([self.initial_state])
    self.S_indicator = self.S > 0
    self.num_infected_neighbors = []
    self.num_infected_and_treated_neighbors = []
    self.Phi = []
    self.current_state = self.S[-1,:]

    counts_for_likelihood_names = ['n_00_0', 'n_01_0', 'n_10_0', 'n_11_0', 'n_00_1', 'n_01_1', 'n_10_1', 'n_11_1',
                                   'a_0', 'a_1']
    self.counts_for_likelihood = {count_name: [] for count_name in counts_for_likelihood_names}

  ##############################################################
  ##            Feature function computation                  ##
  ##############################################################

  def psi_at_location(self, l, raw_data_block, neighbor_order):
    s, a, y = raw_data_block[l, :]
    psi_l = [0]*8
    encoding = int(1*s + 2*a + 4*y)
    psi_l[encoding] = 1

    if neighbor_order == 1:
      psi_neighbors = [0]*8
    elif neighbor_order == 2:
      psi_neighbors = [0]*64

    for lprime in self.adjacency_list[l]:
      s, a, y = raw_data_block[lprime, :]
      first_order_encoding = int(1*s + 2*a + 4*y)
      if neighbor_order == 2:
        for lprime_prime in self.adjacency_list[lprime]:
          if lprime_prime != l:
            s_prime_prime, a_prime_prime, y_prime_prime = raw_data_block[lprime_prime, :]
            second_order_encoding = first_order_encoding + int(8*s_prime_prime + 16*a_prime_prime + 32*y_prime_prime)
            psi_neighbors[second_order_encoding] += 1
      else:
        psi_neighbors[first_order_encoding] += 1
    return np.concatenate((psi_l, psi_neighbors))

  def psi(self, raw_data_block, neighbor_order):
    """
    :param raw_data_block:
    :return:
    """
    if neighbor_order == 1:
      psi = np.zeros((0, 16))
    elif neighbor_order == 2:
      psi = np.zeros((0, 72))

    for l in range(self.L):
      psi_l = self.psi_at_location(l, raw_data_block, neighbor_order)
      psi = np.vstack((psi, psi_l))
    return psi

  def psi_at_action(self, old_raw_data_block, old_data_block, old_action, action, neighbor_order):
    new_data_block = copy.copy(old_data_block)
    if self.add_neighbor_sums:
      new_data_block = new_data_block[:, :int(new_data_block.shape[1] / 2)]
    locations_with_changed_actions = set(np.where(old_action != action)[0])

    for l in range(self.L):
      l_and_neighbors = [l] + self.adjacency_list[l]
      if self.is_any_element_in_set(l_and_neighbors, locations_with_changed_actions):
        new_data_block[l, :] = self.psi_at_location(l, old_raw_data_block, neighbor_order)
    if self.add_neighbor_sums:
      new_data_block = self.psi_neighbor(new_data_block)
    return new_data_block

  def psi_neighbor(self, data_bloc):
    """
    To psi features, concatenate another block of features which are the _sums of the neighboring psi features_.
    :param data_block:
    :return:
    """
    neighbor_data_block = np.zeros(data_block.shape)
    for l in range(self.L):
      neighbor_data_block[l] = np.sum(data_block[self.adjacency_list[l],:], axis=0)
    return np.column_stack((data_block, neighbor_data_block))

  ##############################################################
  ##            End path-based feature function stuff         ##
  ##############################################################

  def add_state(self, s):
    self.S = np.vstack((self.S, s))
    self.S_indicator = np.vstack((self.S_indicator, s > 0))
    self.current_state = s

  def next_state(self):
    """
    Update state array acc to AR(1)
    :return next_state: self.L-length array of new states
    """
    super(SIS, self).next_state()
    next_state = np.random.multivariate_normal(mean=self.beta[0]*self.current_state, cov=self.state_covariance)
    self.add_state(next_state)
    return next_state

  def infection_probability(self, a, y, s, eta=ETA):
    return sis_infection_probability(a, y, eta, self.L, self.adjacency_list, **{'omega': self.omega, 's': s})

  def next_infected_probabilities(self, a, eta=ETA):
    if self.contaminator is not None and self.epsilon > 0:
      current_X_raw_at_action = np.column_stack((self.current_state > 0, a, self.current_infected))
      current_X_at_action = self.psi(current_X_raw_at_action, neighbor_order=1)
      contaminator_probs = self.contaminator.predict_proba(current_X_at_action)
      if self.epsilon == 1.0:
        return contaminator_probs
      else:
        SIS_probs = self.infection_probability(a, self.current_infected, self.current_state, eta=eta)
        return (1 - self.epsilon) * SIS_probs + self.epsilon * contaminator_probs
    else:
      return self.infection_probability(a, self.current_infected, self.current_state, eta=eta)

  def add_infections(self, y):
    self.Y = np.vstack((self.Y, y))
    self.current_infected = y

  def next_infections(self, a, eta=None):
    """
    Updates the vector indicating infections (self.current_infected).
    Computes probability of infection at each state, then generates corresponding
    Bernoullis.
    :param eta:
    :param a: self.L-length binary array of actions at each state
    """
    super(SIS, self).next_infections(a)
    if eta is None:
      next_infected_probabilities = self.next_infected_probabilities(a, eta=self.ETA)
    else:
      next_infected_probabilities = self.next_infected_probabilities(a, eta=eta)
    next_infections = np.random.binomial(n=[1]*self.L, p=next_infected_probabilities)
    self.true_infection_probs.append(next_infected_probabilities)
    self.add_infections(next_infections)

  def neighbor_infection_and_treatment_status(self, l, a, y):
    neighbor_ixs = self.adjacency_list[l]
    num_infected_neighbors = int(np.sum(y[neighbor_ixs]))
    num_treated_and_infected_neighbors = \
      int(np.sum(np.multiply(a[neighbor_ixs], y[neighbor_ixs])))
    num_untreated_and_infected_neighbors = num_infected_neighbors - num_treated_and_infected_neighbors
    return num_treated_and_infected_neighbors, num_untreated_and_infected_neighbors

  def update_counts_for_likelihood(self, data_block, y, y_next):
    """
    Counts of treatment status x neighbor treatment status x neighbor infection status, for fitting SIS model.
    Subscripts denote (location treatment status, neighbor treatment status, next_infection_status)

    :param data_block: psi(raw_data_block) (L x 16-size array)
    :param y_next: next infections after data_block
    :return:
    """
    new_counts_for_likelihood = counts_for_likelihood_at_data_block(data_block, y, y_next)
    self.counts_for_likelihood = update_counts_for_likelihood_(self.counts_for_likelihood, data_block, y, y_next)

  def update_obs_history(self, a):
    """
    :param a: self.L-length array of binary actions at each state
    """
    super(SIS, self).update_obs_history(a)
    raw_data_block = np.column_stack((self.S_indicator[-2,:], a, self.Y[-2,:]))
    data_block_1 = self.psi(raw_data_block, neighbor_order=1)
    data_block_2 = self.psi(raw_data_block, neighbor_order=2)

    # Main features
    self.X_raw.append(raw_data_block)
    self.X.append(data_block_1)
    self.X_2.append(data_block_2)
    self.y.append(self.current_infected)

    # Update likelihood counts
    self.update_counts_for_likelihood(data_block_1, self.Y[-2, :], self.current_infected)

  def data_block_at_action(self, data_block_ix, action, neighbor_order=1, raw=False):
    """
    Replace action in raw data_block with given action.
    """
    super(SIS, self).data_block_at_action(data_block_ix, action)
    if raw:
     new_data_block = copy.copy(self.X_raw[data_block_ix])
     new_data_block[:, 1] = action
    else:
      y = self.Y[data_block_ix, :]
      new_data_block = self.psi(np.column_stack((self.S_indicator[data_block_ix, :], action, y)),
                                neighbor_order)
    return new_data_block

  def raw_data_block_at_action(self, data_block_ix, action):
    """

    :param data_block_ix:
    :param action:
    :return:
    """
    new_raw_data_block = copy.copy(self.X_raw[data_block_ix])
    new_raw_data_block[:, 1] = action
    return new_raw_data_block

  def mb_covariance(self, mb_params):
    """
    Compute covariance of mb estimator, assuming omega=0.

    :param mb_params:
    :return:
    """
    dim = len(mb_params)
    grad_outer = np.zeros((dim, dim))
    hess = np.zeros((dim, dim))

    for t in range(self.T):
      data_block, raw_data_block = self.X[t], self.X_raw[t]
      a, y = raw_data_block[:, 1], raw_data_block[:, 2]
      for l in range(self.L):
        x_raw, x, y_next = raw_data_block[l, :], data_block[l, :], self.y[t][l]

        # MB gradient
        if raw_data_block[l, 2]:
          # Compute gradient of  recovery model
          recovery_features = np.concatenate(([1.0], [a[l]]))
          mb_grad = gradient.logit_gradient(recovery_features, y_next, mb_params[-2:])
          mb_grad = np.concatenate((np.zeros(dim - 2), mb_grad))
          mb_hess = gradient.logit_hessian(recovery_features, mb_params[-2:])
          mb_hess = block_diag(np.zeros((dim - 2, dim - 2)), mb_hess)

        else:
          # Compute gradient of infection model
          num_treated_and_infected_neighbors, num_untreated_and_infected_neighbors = \
            self.neighbor_infection_and_treatment_status(l, a, y)

          def mb_log_lik_at_x(mb_params_infect):
            lik = mb_log_lik_single(mb_params_infect, x_raw, y_next, num_treated_and_infected_neighbors,
                                    num_untreated_and_infected_neighbors)
            return lik

          mb_grad = gradient.central_diff_grad(mb_log_lik_at_x, mb_params[:5])
          mb_grad = np.concatenate((mb_grad, np.zeros(2)))
          mb_hess = gradient.central_diff_hess(mb_log_lik_at_x, mb_params[:5])
          mb_hess = block_diag(mb_hess, np.zeros((2, 2)))

        grad_outer_lt = np.outer(mb_grad, mb_grad)

        grad_outer += grad_outer_lt
        hess += mb_hess

    hess_inv = np.linalg.inv(hess + 0.1 * np.eye(dim))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(self.L * self.T)
    return cov

  def joint_mf_and_mb_covariance(self, mb_params, fitted_mf_clf):
    """
    Compute covariance of mf and mb estimators, where mb_params are maximum likelihood estimate of sis model with
    omega=0, and mf is fitted to env.X using SKLogit2.

    ToDo: This can be optimized!

    :param mb_params:
    :param fitted_mf_clf: flexible SKLogit2-like classifier
    :return:
    """
    mf_params = np.concatenate((fitted_mf_clf.inf_params, fitted_mf_clf.not_inf_params))
    mb_dim = len(mb_params)
    dim = len(mb_params) + len(mf_params)
    grad_outer = np.zeros((dim, dim))
    hess = np.zeros((dim, dim))

    for t in range(self.T):
      data_block, raw_data_block = self.X[t], self.X_raw[t]
      a, y = raw_data_block[:, 1], raw_data_block[:, 2]
      for l in range(self.L):
        x_raw, x, y_next = raw_data_block[l, :], data_block[l, :], self.y[t][l]

        # MB gradient
        if raw_data_block[l, 2]:
          # Compute gradient of  recovery model
          recovery_features = np.concatenate(([1.0], [a[l]]))
          mb_grad = gradient.logit_gradient(recovery_features, y_next, mb_params[-2:])
          mb_grad = np.concatenate((np.zeros(mb_dim-2), mb_grad))
          mb_hess = gradient.logit_hessian(recovery_features, mb_params[-2:])
          mb_hess = block_diag(np.zeros((mb_dim-2, mb_dim-2)), mb_hess)

        else:
          # Compute gradient of infection model
          num_treated_and_infected_neighbors, num_untreated_and_infected_neighbors = \
            self.neighbor_infection_and_treatment_status(l, a, y)

          def mb_log_lik_at_x(mb_params_infect):
            lik = mb_log_lik_single(mb_params_infect, x_raw, y_next, num_treated_and_infected_neighbors,
                                     num_untreated_and_infected_neighbors)
            return lik

          mb_grad = gradient.central_diff_grad(mb_log_lik_at_x, mb_params[:5])
          mb_grad = np.concatenate((mb_grad, np.zeros(2)))
          mb_hess = gradient.central_diff_hess(mb_log_lik_at_x, mb_params[:5])
          mb_hess = block_diag(mb_hess, np.zeros((2, 2)))

        # MF gradient
        mf_features = np.concatenate(([1.0], x))
        mf_grad = fitted_mf_clf.log_lik_gradient(mf_features, y_next, y[l])
        mf_hess = fitted_mf_clf.log_lik_hess(mf_features, y[l])

        # Get gradient and hess for stacked (MB, MF) estimating equation
        grad_lt = np.concatenate((mb_grad, mf_grad))
        grad_outer_lt = np.outer(grad_lt, grad_lt)
        hess_lt = block_diag(mb_hess, mf_hess)

        grad_outer += grad_outer_lt
        hess += hess_lt

    hess_inv = np.linalg.inv(hess + 0.1*np.eye(dim))
    cov = np.dot(hess_inv, np.dot(grad_outer, hess_inv)) / float(self.L * self.T)
    return cov


## Helpers ##
def mb_log_lik_single(mb_params, x_raw, y_next, num_treated_and_infected_neighbors,
                      num_untreated_and_infected_neighbors):
  a = x_raw[1]
  n_0, n_1, n_00, n_11, n_01, n_10 = 1 - a, a, (1-a)*num_untreated_and_infected_neighbors, \
                                           a*num_treated_and_infected_neighbors, \
                                     (1-a)*num_treated_and_infected_neighbors, a*num_untreated_and_infected_neighbors
  eta0 = mb_params[0]
  eta0p1 = eta0 + mb_params[1]
  eta2 = mb_params[2]
  eta2p3 = eta2 + mb_params[3]
  eta2p3p4 = eta2p3 + mb_params[4]
  eta2p4 = eta2 + mb_params[4]

  lik = success_or_failure_component(eta0, eta0p1, eta2, eta2p3, eta2p3p4, eta2p4, np.array([n_00]), np.array([n_01]),
                                     np.array([n_11]), np.array([n_10]), np.array([a]), success=y_next)
  return lik


def counts_for_likelihood_at_data_block(data_block, y, y_next, indices=None):
  """

  :param data_block:
  :param y:
  :param y_next:
  :param indices: Indices at which to compute counts; used for cross-validation train/test split.
  :return:
  """
  if indices is None:
    indices = [l for l in range(data_block.shape[0])]

  treatment_indices = np.array([2, 3, 6, 7])  # Indices corresponding to encodings where a = 1
  neighbor_is_infected_and_treated_indices = np.array([6, 7]) + 8
  neighbor_is_infected_and_not_treated_indices = np.array([4, 5]) + 8

  not_infected_ixs = np.intersect1d(np.where(y == 0)[0], indices)
  X, y_next = data_block[not_infected_ixs], y_next[not_infected_ixs]

  next_infected_ixs = np.where(y_next == 1)
  next_not_infected_ixs = np.where(y_next == 0)
  is_treated = np.sum(X[:, treatment_indices], axis=1)
  num_neighbor_is_treated = np.sum(X[:, neighbor_is_infected_and_treated_indices], axis=1)
  num_neighbor_is_not_treated = np.sum(X[:, neighbor_is_infected_and_not_treated_indices], axis=1)

  n_00 = (1 - is_treated) * num_neighbor_is_not_treated
  n_01 = (1 - is_treated) * num_neighbor_is_treated
  n_10 = is_treated * num_neighbor_is_not_treated
  n_11 = is_treated * num_neighbor_is_treated

  # Need to know whether the next infection status is "success" or "failure"
  n_00_0 = n_00[next_not_infected_ixs]
  n_00_1 = n_00[next_infected_ixs]
  n_01_0 = n_01[next_not_infected_ixs]
  n_01_1 = n_01[next_infected_ixs]
  n_10_0 = n_10[next_not_infected_ixs]
  n_10_1 = n_10[next_infected_ixs]
  n_11_0 = n_11[next_not_infected_ixs]
  n_11_1 = n_11[next_infected_ixs]
  a_0 = is_treated[next_not_infected_ixs].astype(float)
  a_1 = is_treated[next_infected_ixs].astype(float)

  return {'n_00_0': n_00_0, 'n_00_1': n_00_1, 'n_01_0': n_01_0, 'n_01_1': n_01_1, 'n_10_0': n_10_0, 'n_10_1': n_10_1,
          'n_11_0': n_11_0, 'n_11_1': n_11_1, 'a_0': a_0, 'a_1': a_1}


def update_counts_for_likelihood_(counts_for_likelihood, data_block, y, y_next, indices=None):
  new_counts_for_likelihood = counts_for_likelihood_at_data_block(data_block, y, y_next, indices=indices)
  counts_for_likelihood['n_00_0'] = np.append(counts_for_likelihood['n_00_0'],
                                                   new_counts_for_likelihood['n_00_0'])
  counts_for_likelihood['n_00_1'] = np.append(counts_for_likelihood['n_00_1'],
                                                   new_counts_for_likelihood['n_00_1'])
  counts_for_likelihood['n_01_0'] = np.append(counts_for_likelihood['n_01_0'],
                                                   new_counts_for_likelihood['n_01_0'])
  counts_for_likelihood['n_01_1'] = np.append(counts_for_likelihood['n_01_1'],
                                                   new_counts_for_likelihood['n_01_1'])
  counts_for_likelihood['n_10_0'] = np.append(counts_for_likelihood['n_10_0'],
                                                   new_counts_for_likelihood['n_10_0'])
  counts_for_likelihood['n_10_1'] = np.append(counts_for_likelihood['n_10_1'],
                                                   new_counts_for_likelihood['n_10_1'])
  counts_for_likelihood['n_11_0'] = np.append(counts_for_likelihood['n_11_0'],
                                                   new_counts_for_likelihood['n_11_0'])
  counts_for_likelihood['n_11_1'] = np.append(counts_for_likelihood['n_11_1'],
                                                   new_counts_for_likelihood['n_11_1'])
  counts_for_likelihood['a_0'] = np.append(counts_for_likelihood['a_0'],
                                                new_counts_for_likelihood['a_0'])
  counts_for_likelihood['a_1'] = np.append(counts_for_likelihood['a_1'],
                                                new_counts_for_likelihood['a_1'])
  return counts_for_likelihood