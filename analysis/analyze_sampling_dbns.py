import numpy as np
import yaml

NETWORK_NAMES = ['lattice', 'nearestneighbor']


def summarize_sampling_dbns(fname_list):
  results_dict = {'L': [], 'policy': [], 'network': [], 'coverages': [], 'min_pvals': []}
  for fname in fname_list:
    d = yaml.load(open('./results/{}'.format(fname), 'rb'))

    # Get summaries of sampling dbn comparison with bootstrap dbns
    coverages = d['results']['coverages'] # Coverages of bootstrap naive conf intervals
    bootstrap_pvals = d['results']['bootstrap_pvals'] # Pvals of ks comparisons of bootstrap and sampling dbns
    min_bootstrap_vals = np.array(bootstrap_pvals).min(axis=0)

    # Get gen model settings
    L = d['settings']['L']
    policy = d['settings']['policy_name']
    for name in NETWORK_NAMES:
      if name in fname:
        network_name = name

    # Add to results
    results_dict['L'].append(L)
    results_dict['policy'].append(policy)
    results_dict['network'].append(network_name)
    results_dict['coverages'].append(coverages)
    results_dict['min_pvals'].append(min_bootstrap_vals)





