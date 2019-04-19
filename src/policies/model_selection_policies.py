import os
this_dir = os.path.dirname(os.path.abspath(__file__))

from src.estimation.q_functions.fqi import fqi
from src.estimation.q_functions.regressor import AutoRegressor
from src.estimation.q_functions.q_functions import q, q_max_all_states
from src.estimation.model_based.sis.estimate_sis_q_fn import estimate_sis_q_fn
from src.estimation.model_based.sis.estimate_sis_parameters import fit_sis_transition_model, fit_infection_prob_model
from src.estimation.model_based.Gravity.estimate_ebola_parameters import fit_ebola_transition_model
from src.estimation.model_based.Gravity.estimate_continuous_parameters import fit_continuous_grav_transition_model
from src.estimation.q_functions.model_fitters import SKLogit2
from src.policies.policy_search import policy_search, features_for_priority_score
from src.policies.q_function_policies import two_step
import src.estimation.q_functions.mse_optimal_combination as mse_combo
from src.estimation.q_functions.one_step import *
from src.utils.misc import random_argsort
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from scipy.special import expit, logit
import numpy as np
import keras.backend as K
from functools import partial


def sis_aic_two_step(**kwargs):
  """
  Use AIC to select model-free or model-based probability estimator.  If model-based is selected, use policy search.
  If model-free is selected, use two-step.

  :param kwargs:
  :return:
  """
  mf_classifier, env = kwargs['classifier'], kwargs['env']

  # Get aic of model-free and model-based estimators
  mf_classifier.fit(env.vstack(env.X), env.hstack(env.y))
  mf_aic = mf_classifier.aic
  beta_mean, mb_aic = fit_infection_prob_model(env, None)

  if mf_aic < mb_aic:  # Model-free policy
    return two_step(**kwargs)
  else: # Model-based policy search
    treatment_budget, remaining_time_horizon, initial_policy_parameter = \
      kwargs['treatment_budget'], kwargs['planning_depth'], kwargs['initial_policy_parameter']
    if env.__class__.__name__ == "SIS":
      beta_mean, _ = fit_infection_prob_model(env, None)
      beta_cov = env.mb_covariance(beta_mean)

      def gen_model_posterior():
        beta_tilde = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)
        return beta_tilde
    elif env.__class__.__name__ == "Gravity":
      # beta_mean = fit_ebola_transition_model(env)
      # beta_cov = env.mb_covariance(beta_mean)
      def gen_model_posterior():
        beta_tilde = fit_ebola_transition_model(env, bootstrap=True)
        return beta_tilde

    # Settings
    if initial_policy_parameter is None:
      initial_policy_parameter = np.ones(3) * 0.5
    initial_alpha = initial_zeta = None
    # remaining_time_horizon = T - env.T

    # ToDo: These were tuned using bayes optimization on 10 mc replicates from posterior obtained after 15 steps of random
    # ToDo: policy; may be improved...
    rho = 3.20
    tau = 0.76
    a, policy_parameter = policy_search(env, remaining_time_horizon, gen_model_posterior, initial_policy_parameter,
                                        initial_alpha, initial_zeta, treatment_budget, rho, tau, tol=1e-3,
                                        maxiter=100, feature_function=features_for_priority_score, k=1,
                                        method='bayes_opt')
    return a, {'initial_policy_parameter': policy_parameter}




