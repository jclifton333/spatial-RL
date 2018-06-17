# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:06:36 2018

@author: Jesse
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import networkx as nx
import pdb

def polynomialFeatures(num_raw_features, interaction_only):
  poly = PolynomialFeatures(interaction_only=interaction_only)
  dummy = np.zeros((1, num_raw_features))
  poly.fit_transform(dummy)
  return poly.transform

  
# Path features
def get_all_paths_from_node(graph, node_ix, path_length):
  result = []
  all_simple_paths = (nx.all_simple_paths(graph, node_ix, target, path_length) for target in graph.nodes()
                      if target > node_ix)
  for paths in all_simple_paths:
    result += paths
  return result


def get_all_paths(adjacency_matrix, path_length):
  g = nx.from_numpy_matrix(adjacency_matrix)
  L = adjacency_matrix.shape[0]
  list_of_path_lists = [l for n in range(L) for l in get_all_paths_from_node(g, n, path_length)]
  list_of_path_lists += [[l] for l in range(L)]
  return list_of_path_lists


  
  