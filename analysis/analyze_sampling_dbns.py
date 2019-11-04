import numpy as np
import yaml
import pdb

NETWORK_NAMES = ['lattice', 'nearestneighbor']
INDICES_TO_KEEP = [i for i in range(33) if i not in [17, 18, 19, 20]]


def summarize_sampling_dbns(fname_list, save=False):
  # ToDo: Indices of 0-step q-function that are acting weird are [17, 18, 19, 20]
  # I think these correspond to infection times
  # 17: (s, ~a, ~y)
  # 18: (~s, a, ~y)
  # 19: (s, a, ~y)
  # 20: (~s, ~a, y)
  # which are redundant with features in the main effect component

  results_dict = {'L': [], 'policy': [], 'network': [], 'coverages': [], 'bootstrap_pvals': []}
  for fname in fname_list:
    d = yaml.load(open('./results/{}'.format(fname), 'rb'))

    # Get summaries of sampling dbn comparison with bootstrap dbns
    coverages = d['results']['coverages'] # Coverages of bootstrap naive conf intervals
    bootstrap_pvals = d['results']['bootstrap_pvals'] # Pvals of ks comparisons of bootstrap and sampling dbns
    mean_bootstrap_pvals = np.array(bootstrap_pvals).mean(axis=0)
    coverages = [coverages[ix] for ix in range(len(coverages)) if ix in INDICES_TO_KEEP] 
    mean_bootstrap_pvals = [mean_bootstrap_pvals[ix] for ix in range(len(mean_bootstrap_pvals)) if ix in INDICES_TO_KEEP] 

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
    results_dict['bootstrap_pvals'].append(bootstrap_pvals)

  if save: 
    savename = 'sampling-dbn-results.yml'
    with open(savename, 'w') as outfile:
      yaml.dump(results_dict, outfile)
  return 


if __name__ == "__main__":
  fname_list_ = ['sis_true_probs_myopic_random_100_nearestneighbor_sampling-dbn-run=True_0.0_191103_213222.yml', 
                 'sis_true_probs_myopic_random_25_nearestneighbor_sampling-dbn-run=True_0.0_191103_212511.yml', 
                 'sis_true_probs_myopic_random_300_nearestneighbor_sampling-dbn-run=True_0.0_191103_214306.yml', 
                 'sis_true_probs_myopic_random_50_nearestneighbor_sampling-dbn-run=True_0.0_191103_212808.yml',
                 'sis_true_probs_myopic_random_100_lattice_sampling-dbn-run=True_0.0_191104_110012.yml',
                 'sis_true_probs_myopic_random_25_lattice_sampling-dbn-run=True_0.0_191104_105124.yml',
                 'sis_true_probs_myopic_random_300_lattice_sampling-dbn-run=True_0.0_191104_111133.yml',
                 'sis_true_probs_myopic_random_50_lattice_sampling-dbn-run=True_0.0_191104_105510.yml']
  summarize_sampling_dbns(fname_list_, save=True)





