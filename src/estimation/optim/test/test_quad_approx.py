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
from copy import copy
from itertools import combinations

TRUE_MS = [np.array([[1, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]),
           np.array([[0, 0, 0, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]),
           np.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0]]),
           np.array([[0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0],
                     [0, 0, 0, 2]])]


def get_sample_q_and_act():
  sample_acts = []
  sample_qs = []
  ixs = [c for i in range(4) for c in combinations(range(4), i)] + [(0, 1, 2, 3)]
  for ix in ixs:
    dummy_act = np.zeros(4)
    dummy_act[list(ix)] = 1
    a = copy(dummy_act)
    sample_acts.append(a)
    sample_qs.append([np.dot(a, np.dot(M, a)) for M in TRUE_MS])
  return np.array(sample_qs), sample_acts


def get_adjacency_and_neighbor_interaction_lists():
  adjacency_list = [[1,2], [0,3], [0,3], [1,2]]
  neighbor_interaction_lists = [np.array([[i,j] for i in [l] + adjacency_list[l] for j in [l] + adjacency_list[l] if j >= i])
                                for l in range(4)]
  return adjacency_list, neighbor_interaction_lists


def test_answer():
  sample_qs, sample_acts = get_sample_q_and_act()
  adjacency_list, neighbor_interaction_lists = get_adjacency_and_neighbor_interaction_lists()
  q, i = fit_quad_approx(sample_qs, sample_acts, neighbor_interaction_lists, 4)
  assert np.allclose(q, np.sum(TRUE_MS, axis=0))

