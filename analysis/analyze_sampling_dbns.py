import numpy as np
import yaml
import pdb
import pandas as pd
from scipy.stats import pearsonr

NETWORK_NAMES = ['lattice', 'nearestneighbor']
INDICES_TO_KEEP = [i for i in range(33) if i not in [17, 18, 19, 20]]


def summarize_sampling_dbns(fname_list, outname=None, save=False):
  # ToDo: Indices of 0-step q-function that are acting weird are [17, 18, 19, 20]
  # I think these correspond to infection times
  # 17: (s, ~a, ~y)
  # 18: (~s, a, ~y)
  # 19: (s, a, ~y)
  # 20: (~s, ~a, y)
  # which are redundant with features in the main effect component

  results_dict = {'L': [], 'policy': [], 'network': [], 'min_coverages': [], 'median_coverages': [], 
                  'max_coverages': [], 'nonzero_corr': []}
  for fname in fname_list:
    d = yaml.load(open('./results/{}'.format(fname), 'rb'))

    # Get summaries of sampling dbn comparison with bootstrap dbns
    coverages = d['results']['coverages'] # Coverages of bootstrap naive conf intervals
    bootstrap_pvals = d['results']['bootstrap_pvals'] # Pvals of ks comparisons of bootstrap and sampling dbns
    mean_bootstrap_pvals = np.array(bootstrap_pvals).mean(axis=0)
    coverages = [coverages[ix] for ix in range(len(coverages)) if ix in INDICES_TO_KEEP] 
    mean_bootstrap_pvals = [mean_bootstrap_pvals[ix] for ix in range(len(mean_bootstrap_pvals)) if ix in INDICES_TO_KEEP] 

    # Get gen model settings
    # ToDo: store these in results so we dont have to do this hacky shit
    L = d['settings']['L']
    policy = (len(d['results']['coverages']) == 33)
    for name in NETWORK_NAMES:
      if name in fname:
        network_name = name
    if 'true_probs_myopic' in fname:
      policy_name = 'true_probs_myopic'
    else:
      policy_name = 'random'

    # Add to results
    results_dict['L'].append(L)
    results_dict['policy'].append(policy_name)
    results_dict['network'].append(network_name)
    results_dict['min_coverages'].append(np.min(coverages))
    results_dict['median_coverages'].append(np.median(coverages))
    results_dict['nonzero_corr'].append(pearsonr(d['results']['mean_counts'], coverages)[0])
    results_dict['max_coverages'].append(np.max(coverages))
    # results_dict['bootstrap_pvals'].append(mean_bootstrap_pvals)

  df = pd.DataFrame(results_dict)
  df.sort_values(by=['network', 'policy', 'L'], inplace=True)
  print(df)
  if save: 
    savename = 'sampling-dbn-results-{}.yml'.format(outname)
    with open(savename, 'w') as outfile:
      yaml.dump(results_dict, outfile)
  return 


if __name__ == "__main__":
  fname_list_1_step_cutoff_nonzero_corr_no_ridge = []

  summarize_sampling_dbns(fname_list_1_step_cutoff_nonzero_corr_no_ridge, outname=None, save=False)





