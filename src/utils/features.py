# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:06:36 2018

@author: Jesse
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import networkx as nx

def polynomialFeatures(num_raw_features, interaction_only):
  poly = PolynomialFeatures(interaction_only=interaction_only)
  dummy = np.zeros((1, num_raw_features))
  poly.fit_transform(dummy)
  return poly.transform  
  
# Path features
def get_all_paths_from_node(graph, node_ix, path_length):
  result = []
  for paths in (nx.all_simple_paths(graph, n, target, l) for target in graph.nodes()):
    result += paths
  return result

def get_all_paths(adjacency_matrix, path_length):
  g = nx.from_numpy_matrix(adjacency_matrix)
  list_of_path_lists = [get_all_paths_from_node(g, n, path_length)
                        for n in range(adjacency_matrix.shape[0])]
  return list_of_path_lists


  
  