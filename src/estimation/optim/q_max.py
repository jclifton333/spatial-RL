import numpy as np
from src.estimation.q_functions.q_functions import q


def q_max_all_states(env, evaluation_budget, treatment_budget, predictive_model, argmaxer,
                     network_features=False):
  """
  Take q_max for all data blocks in env history (need this for Q-learning/rollout).
  """
  q_max_list = []
  for t in range(env.T):
    q_fn = lambda a: q(a, t, env, predictive_model, network_features=network_features)
    argmax = argmaxer(q_fn, evaluation_budget, treatment_budget, env)
    q_max_list.append(q_fn(argmax))
  return np.array(q_max_list), argmax
