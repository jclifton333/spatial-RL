import pdb
import numpy as np
from src.environments.SIS import SIS
from src.environments.generate_network import lattice


def test_phi_0():
  env = SIS(lambda x: x, 4, 0, lattice)
  data_block = np.zeros((4, 3))
  env.map_to_path_signature = {}
  ans = env.phi_k(1, data_block)
  assert np.array_equal(ans, np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), (4, 1)))


def test_phi_at_action():
  env = SIS(lambda x: x, 4, 0, lattice)
  # map_to_path_signature needed to call phi_at_action
  env.map_to_path_signature = {}
  old_data_block = np.zeros((4, 3))
  for k, r_list in env.dict_of_path_lists.items():
    for r in r_list:
      env.map_to_path_signature[r] = env.m_r(r, old_data_block)
  new_data_block = np.zeros((4, 3))
  new_data_block[0, 1] = 1
  correct_ans = env.phi_k(2, new_data_block)
  ans = env.phi_at_action(env.phi(old_data_block), np.array([0, 0, 0, 0]), np.array([1, 0, 0, 0]))
  ans = ans[:, 9:]
  assert np.array_equal(ans, correct_ans)
