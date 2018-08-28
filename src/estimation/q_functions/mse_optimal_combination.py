import pdb
import numpy as np
from .one_step import fit_one_step_mf_and_mb_qs, fit_one_step_predictor
from .model_fitters import SKLogit2
from src.estimation.q_functions.q_functions import q_max_all_states
from sklearn.linear_model import LinearRegression


def softhresholder(x, threshold):
  if -threshold < x < threshold:
    return 0
  else:
    return x - np.sign(x)*threshold


def mse_optimal_convex_combo(bias_1, bias_2, var_1, var_2, covariance):
  mse_1 = bias_1**2 + var_1
  mse_2 = bias_2**2 + var_2
  return (mse_2 - covariance) / (mse_1 + mse_2 - 2*covariance)


def one_step_sis_convex_combo(env):
  """
  Get estimated optimal convex combination of one-step sis mb and mf q-functions.

  :param env:
  :return:
  """
  q_mb, q_mf, mb_params, fitted_mf_clf = fit_one_step_mf_and_mb_qs(env, SKLogit2)

  print('eta hat: {}\neta: {}'.format(mb_params, env.ETA))
  # Compute covariances
  cov = env.joint_mf_and_mb_covariance(mb_params, fitted_mf_clf)

  # Simulate from estimated sampling dbns
  params = np.concatenate((mb_params, fitted_mf_clf.inf_params, fitted_mf_clf.not_inf_params))
  simulated_params = np.random.multivariate_normal(mean=params, cov=cov, size=100)

  yhat_mb_draws = np.zeros((0, env.T * env.L))
  yhat_mf_draws = np.zeros((0, env.T * env.L))

  for simulated_param in simulated_params:
    simulated_mb_params = simulated_param[:len(mb_params)]
    simulated_mf_params = simulated_param[len(mb_params):]
    yhat_mb_draw = np.hstack([env.infection_probability(data_block[:, 1], data_block[:, 2], data_block[:, 0],
                                                        eta=simulated_mb_params) for data_block in env.X_raw])
    yhat_mf_draw = np.hstack([fitted_mf_clf.predict_proba_given_parameter(data_block, np.where(raw_data_block[:, 2] == 1),
                                                                          np.where(raw_data_block[:, 2] == 0),
                                                                      simulated_mf_params)[:, -1]
                          for data_block, raw_data_block in zip(env.X, env.X_raw)])
    yhat_mb_draws = np.vstack((yhat_mb_draws, yhat_mb_draw))
    yhat_mf_draws = np.vstack((yhat_mf_draws, yhat_mf_draw))

  # Compute variances and covariance
  mb_var = np.mean(np.var(yhat_mb_draws.flatten(), axis=0))
  mf_var = np.mean(np.var(yhat_mf_draws.flatten(), axis=0))
  # mb_mf_cov = np.cov(np.array([np.mean(yhat_mb_draws.flatten(), axis=1), np.mean(yhat_mf_draws.flatten(),
  #                                                                                axis=1)]))[0, 1]

  # Compute bias
  yhat_mb = np.hstack([env.infection_probability(data_block[:, 1], data_block[:, 2], data_block[:, 0],
                                                 eta=mb_params) for data_block in env.X_raw])
  yhat_mf = np.hstack([fitted_mf_clf.predict_proba_given_parameter(data_block, np.where(raw_data_block[:, 2] == 1),
                                                                   np.where(raw_data_block[:, 2] == 0),
                                                                   params[len(mb_params):])[:, -1]
                       for data_block, raw_data_block in zip(env.X, env.X_raw)])
  mb_bias = np.mean(yhat_mb - np.hstack(env.y))
  mf_bias = np.mean(yhat_mf - np.hstack(env.y))

  # Get mixing weight
  alpha_mf = mse_optimal_convex_combo(mf_bias, mb_bias, mf_var, mb_var, 0.0)
  alpha_mb = 1 - alpha_mf

  # We return yhat_mf and _mb to compute variance of higher-order backups
  yhat_mb_draws = [yhat_mb_draw.reshape((env.T, env.L)) for yhat_mb_draw in yhat_mb_draws]
  return alpha_mb, alpha_mf, q_mb, q_mf, yhat_mf_draws, yhat_mb_draws, simulated_params


def two_step_sis_convex_combo(env, gamma, argmaxer, evaluation_budget, treatment_budget,
                              num_parametric_bootstrap_replicates=100):
  """
  Get (estimated) mse-optimal convex combo of 2-step mf and mb q functions, using parametric bootstrap to compute
  variances.
  """

  # Fit consistent infection and state models for parametric bootstrapping
  mf_one_step_predictor, _ = fit_one_step_predictor(SKLogit2, env, None)
  phat_list = [mf_one_step_predictor(data_block, np.where(data_block[:, -1] == 1)[0],
                                     np.where(data_block[:, -1] == 0)[0]) for data_block in env.X]
  S, Sp1 = env.S[:-1].flatten(), env.S[1:].flatten()
  reg = LinearRegression()
  reg.fit(S, Sp1)
  Sp1_mean_list = [reg.predict(s) for s in env.S]
  sigma_hat = np.sqrt(np.sum((Sp1 - Sp1_mean)**2) / (len(S) - 1))

  # Estimate variance with parametric bootstrap
  mb_backup_draws = np.zeros((0, env.T * env.L))
  mf_backup_draws = np.zeros((0, env.T * env.L))
  for rep in range(num_parametric_bootstrap_replicates):
    # Bootstrap draw from fitted model
    Sp1_indicator_draw_list = [np.random.normal(loc=Sp1_mean, scale=sigma_hat) > 0 for Sp1_mean in Sp1_mean_list]
    yp1_draw_list = [np.random.binomial(1, p=phat) for phat in phat_list]

    # Fit 2-step mf and mb on draws
    q_mb, q_mf, mb_params, fitted_mf_clf = fit_one_step_mf_and_mb_qs(env, SKLogit2, y_next=yp1_draw_list.flatten())
    mb_backup_draw = sis_mb_backup(env, gamma, q_mb, q_mb, argmaxer, evaluation_budget, treatment_budget)
    q_mf_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, fitted_mf_clf.predict, argmaxer,
                                      list_of_infections_and_states=zip(Sp1_indicator_draw_list, yp1_draw_list))
    mf_backup_draw = yp1_draw_list.flatten() + gamma * q_mf_max.flatten()

    # Add to array
    mb_backup_draws = np.vstack((mb_backup_draws, mb_backup_draw))
    mf_backup_draws = np.vstack((mf_backup_draws, mf_backup_draw))

  mb_var = np.mean(np.var(mb_backup_draws, axis=0))
  mf_var = np.mean(np.var(mf_backup_draws, axis=0))

  # Estimate bias
  q_mb, q_mf, mb_params, fitted_mf_clf = fit_one_step_mf_and_mb_qs(env, SKLogit2)
  mb_backup = sis_mb_backup(env, gamma, q_mb, q_mb, argmaxer, evaluation_budget, treatment_budget)
  q_mf_max, _, _ = q_max_all_states(env, evaluation_budget, treatment_budget, fitted_mf_clf.predict, argmaxer)
  mf_backup = np.hstack(env.y) + gamma * q_mf_max.flatten()
  mb_bias = np.mean(mb_backup_draws, axis=0) - mb_backup
  mf_bias = np.mena(mf_backup_draws, axis=0) - mf_backup

  # Get estimated optimal mixing weight
  alpha_mf = mse_optimal_convex_combo(mf_bias, mb_bias, mf_var, mb_var, 0.0)
  alpha_mb = 1 - alpha_mf

  return alpha_mb*mb_backup + alpha_mf*mf_backup






def sis_mb_backup(env, gamma, q_mb_one_step, q_mb, argmaxer, evaluation_budget, treatment_budget,
                  number_of_draws=10, phat_list=None):
  if phat_list is None:
    phat_list = [q_mb_one_step(data_block) for data_block in env.X_raw]
  backup = np.zeros((0, env.T * env.L))
  for draw in range(number_of_draws):
    backups_for_draw = np.zeros(0)
    for t in range(env.T):
      phat_t = phat_list[t]
      y_next = np.random.binomial(1, p=phat_t)
      q_fn = lambda a: q_mb(np.column_stack((np.zeros(len(a)), a, y_next)))
      argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env)
      q_max = q_fn(argmax)
      backup_draw = phat_t + gamma * q_max
      backups_for_draw = np.append(backups_for_draw, backup_draw)
    backup = np.vstack((backup, backups_for_draw))
  return np.mean(backup, axis=0)




# def estimate_mb_bias_and_variance(phat, env):
#   """
#
#   :param phat: Estimated one step probabilities
#   :param env:
#   :return:
#   """
#   NUMBER_OF_BOOTSTRAP_SAMPLES = 100
#   THRESHOLD = 0.05  # For softhresholding bias estimate
#
#   mb_bias = 0.0
#   phat_mean_replicates = []
#   for b in range(NUMBER_OF_BOOTSTRAP_SAMPLES):
#     one_step_q_b = fit_one_step_mb_q(env, np.random.multinomial(env.L, np.ones(env.L)/env.L, size=env.T))
#     phat_b = np.hstack([one_step_q_b(data_block) for data_block in env.X])
#     mb_bias += (np.mean(phat - phat_b) - mb_bias) / (b + 1)
#     phat_mean_replicates.append(np.mean(phat_b))
#   # mb_variance = np.var(phat_mean_replicates)
#
#   mb_variance = estimate_mb_variance(phat, env)
#
#   print('bias before threshold: {}'.format(mb_bias))
#   mb_bias = softhresholder(mb_bias, THRESHOLD)
#   return mb_bias, mb_variance
#
#
# def estimate_mb_variance(phat, env):
#   # Get mb variance from asymptotic normality
#   X_raw = np.vstack(env.X_raw)
#   V = np.diag(np.multiply(phat, 1 - phat))
#   beta_hat_cov_inv = np.dot(X_raw.T, np.dot(V, X_raw))
#   beta_hat_cov = np.linalg.inv(beta_hat_cov_inv + 0.01*np.eye(beta_hat_cov_inv.shape[0]))
#   X_beta_hat_cov = np.dot(X_raw, np.dot(beta_hat_cov, X_raw.T))
#   mb_variance = np.sum(np.multiply(np.diag(X_beta_hat_cov), gradient.expit_derivative(phat)**2)) / len(phat)**2
#   return mb_variance


# def estimate_mf_variance(phat):
#   """
#   Parametric estimate of one step mf variance, using one step mb.
#   :param phat: estimated one step probabilities
#   :param env:
#   """
#   mf_variance = np.multiply(phat, 1 - phat)
#   return np.sum(mf_variance) / len(mf_variance)**2


# def estimate_alpha_from_mse_components(mb_bias, mb_variance, mf_variance, covariance):
#   """
#
#
#   :param mb_bias:
#   :param mb_variance:
#   :param mf_variance:
#   :param covariance: estimated covariance of mf and mb estimators
#   :return:
#   """
#   # Estimate alpha
#   alpha_mf = (mb_variance + mb_bias**2 - covariance) / (mb_bias**2 + mb_variance + mf_variance - 2*covariance)
#
#   # Clip to [0, 1]
#   alpha_mf = np.max((0.0, np.min((1.0, alpha_mf))))
#   alpha_mb = 1 - alpha_mf
#   return alpha_mb, alpha_mf
#
#
# def estimate_mse_optimal_convex_combination(q_mb_one_step, env):
#   """
#   Get estimated mse-optimal convex combination weight for combining q-functions
#   """
#
#   phat = np.hstack([q_mb_one_step(data_block) for data_block in env.X_raw])
#   mb_bias, mb_variance = estimate_mb_bias_and_variance(phat, env)
#   covariance = 0.0
#   mf_variance = estimate_mf_variance(phat)
#   mb_variance = np.min((mb_variance, mf_variance))  # We know that mb_variance must be smaller than mf_variance
#   alpha_mb, alpha_mf = estimate_alpha_from_mse_components(phat, mb_bias, mb_variance, mf_variance, covariance)
#   mse_components = {'mb_bias': mb_bias, 'mb_variance': mb_variance, 'mf_variance': mf_variance,
#                     'covariance': covariance}
#   return alpha_mb, alpha_mf, phat, mse_components