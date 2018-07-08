from .greedy_gq import ggq


def stack(q_1, q_2, gamma, holdout_env, evaluation_budget, treatment_budget, argmaxer, intercept=False):
  theta = ggq([q_1, q_2], gamma, holdout_env, evaluation_budget, treatment_budget, argmaxer, intercept, project=True)
  return theta

