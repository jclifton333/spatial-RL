"""
We'll test quad_approx for a 2x2 lattice with the following q function:

q_0(a) = a_0 + a_0*a_1
q_1(a) = a_1 + a_1*a_2
q_2(a) = a_2 + a2*a_3
q_3(a) = 2*a_3 + a_3*a_0

and a treatment budget of 1.
"""
import numpy as np
from src.estimation.optim.quad_approx.fit_quad_approx import fit_quad_approx


def get_sample_q_and_act():
  sample_acts = []
  for i in range(4):
    dummy_act = np.zeros(4)
    dummy_act[i] = 1
    sample_acts.append(dummy_act)
  sample_q = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0],
                      [0,0,0,2]])
  return sample_q, sample_acts


def get_adjacency_and_neighbor_interaction_lists():
  adjacency_list = [[1,2], [0,3], [1,3], [1,2]]
  neighbor_interaction_lists = [np.array([[i,j] for i in adjacency_list[l] for j in adjacency_list[l]])
                                for l in range(4)]
  return adjacency_list, neighbor_interaction_lists


def get_correct_parameters():
  linear_parameter = np.array([1,1,1,2])
  quadratic_parameter = np.array([[0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0]])
  intercept = 0
  return linear_parameter, quadratic_parameter, intercept


def test_answer():
  sample_qs, sample_acts = get_sample_q_and_act()
  adjacency_list, neighbor_interaction_lists = get_adjacency_and_neighbor_interaction_lists()
  linear_parameter, quadratic_parameter, intercept = get_correct_parameters()
  q, l, i = fit_quad_approx(sample_qs, sample_acts, adjacency_list, neighbor_interaction_lists, 4)
  assert np.array_equal(q, quadratic_parameter)

