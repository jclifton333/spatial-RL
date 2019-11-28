import numpy as np
import yaml
import pdb
import pandas as pd

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
                  'max_coverages': []}
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
  fname_list_0_step = ['sis_true_probs_myopic_random_100_nearestneighbor_sampling-dbn-run=True_0.0_191103_213222.yml',
                 'sis_true_probs_myopic_random_25_nearestneighbor_sampling-dbn-run=True_0.0_191103_212511.yml', 
                 'sis_true_probs_myopic_random_300_nearestneighbor_sampling-dbn-run=True_0.0_191103_214306.yml', 
                 'sis_true_probs_myopic_random_50_nearestneighbor_sampling-dbn-run=True_0.0_191103_212808.yml',
                 'sis_true_probs_myopic_random_100_lattice_sampling-dbn-run=True_0.0_191104_110012.yml',
                 'sis_true_probs_myopic_random_25_lattice_sampling-dbn-run=True_0.0_191104_105124.yml',
                 'sis_true_probs_myopic_random_300_lattice_sampling-dbn-run=True_0.0_191104_111133.yml',
                 'sis_true_probs_myopic_random_50_lattice_sampling-dbn-run=True_0.0_191104_105510.yml',
                 'sis_random_random_100_nearestneighbor_sampling-dbn-run=True_0.0_191104_152354.yml', 
                 'sis_random_random_25_nearestneighbor_sampling-dbn-run=True_0.0_191104_151444.yml', 
                 'sis_random_random_25_lattice_sampling-dbn-run=True_0.0_191104_145243.yml', 
                 'sis_random_random_300_lattice_sampling-dbn-run=True_0.0_191104_151227.yml', 
                 'sis_random_random_300_nearestneighbor_sampling-dbn-run=True_0.0_191104_153606.yml',
                 'sis_random_random_50_lattice_sampling-dbn-run=True_0.0_191104_145633.yml', 
                 'sis_random_random_50_nearestneighbor_sampling-dbn-run=True_0.0_191104_151837.yml', 
                 'sis_random_random_100_lattice_sampling-dbn-run=True_0.0_191104_150118.yml', 
                 'sis_random_random_1000_lattice_sampling-dbn-run=True_0.0_191105_200724.yml', 
                 'sis_random_random_1000_nearestneighbor_sampling-dbn-run=True_0.0_191105_204542.yml', 
		 'sis_random_random_500_nearestneighbor_sampling-dbn-run=True_0.0_191105_201740.yml', 
		 'sis_random_random_500_lattice_sampling-dbn-run=True_0.0_191105_193149.yml', 
		 'sis_true_probs_myopic_random_1000_lattice_sampling-dbn-run=True_0.0_191105_192112.yml', 
		 'sis_true_probs_myopic_random_1000_nearestneighbor_sampling-dbn-run=True_0.0_191105_212810.yml', 
		 'sis_true_probs_myopic_random_500_lattice_sampling-dbn-run=True_0.0_191105_185341.yml', 
		 'sis_true_probs_myopic_random_500_nearestneighbor_sampling-dbn-run=True_0.0_191105_205547.yml'
                 ]

  # ToDo: double check that these are all 1-step (not 0-step)
  fname_list_1_step_max = ['sis_true_probs_myopic_random_50_lattice_sampling-dbn-run=True_0.0_191104_170925.yml',
                 'sis_true_probs_myopic_random_300_lattice_sampling-dbn-run=True_0.0_191104_174921.yml',
                 'sis_true_probs_myopic_random_25_lattice_sampling-dbn-run=True_0.0_191104_170226.yml',
                 'sis_true_probs_myopic_random_100_lattice_sampling-dbn-run=True_0.0_191104_172008.yml',
                 'sis_random_random_1000_lattice_sampling-dbn-run=True_0.0_191105_145035.yml', 
		 'sis_random_random_500_lattice_sampling-dbn-run=True_0.0_191105_130059.yml', 
                 'sis_random_random_500_nearestneighbor_sampling-dbn-run=True_0.0_191105_154222.yml', 
                 'sis_true_probs_myopic_random_1000_lattice_sampling-dbn-run=True_0.0_191105_122930.yml',
                 'sis_true_probs_myopic_random_500_lattice_sampling-dbn-run=True_0.0_191105_111508.yml',
                 'sis_true_probs_myopic_random_500_nearestneighbor_sampling-dbn-run=True_0.0_191105_172232.yml',
                 'sis_true_probs_myopic_random_1000_nearestneighbor_sampling-dbn-run=True_0.0_191105_183232.yml',
                 'sis_random_random_1000_nearestneighbor_sampling-dbn-run=True_0.0_191105_165429.yml']

  fname_list_1_step_random = ['sis_true_probs_myopic_random_25_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191107_144645.yml', 
                              'sis_true_probs_myopic_random_50_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191107_182922.yml', 
                              'sis_true_probs_myopic_random_300_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191108_092426.yml',
                              'sis_random_random_50_nearestneighbor_sampling-dbn-run=True_eval=two_step_random_0.0_191109_080552.yml', 
                              'sis_random_random_300_nearestneighbor_sampling-dbn-run=True_eval=two_step_random_0.0_191110_023608.yml', 
                              'sis_true_probs_myopic_random_50_nearestneighbor_sampling-dbn-run=True_eval=two_step_random_0.0_191110_054358.yml',
                              'sis_true_probs_myopic_random_500_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191111_175443.yml',
                              'sis_true_probs_myopic_random_1000_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191115_055426.yml',
                              'sis_random_random_1000_lattice_sampling-dbn-run=True_eval=two_step_random_0.0_191120_071226.yml']

  fname_list_1_step_cutoff = ['sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191124_225649.yml',
                              'sis_random_quad_approx_50_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191124_224915.yml',
                              'sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_003117.yml',
                              'sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_003117.yml', 
                              'sis_random_quad_approx_1000_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_173330.yml', 
                              'sis_random_quad_approx_100_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_155347.yml',
                              'sis_random_quad_approx_2000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_154038.yml', 
                              'sis_random_quad_approx_50_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191125_154535.yml',
                              'sis_true_probs_myopic_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_062513.yml',
                              'sis_true_probs_myopic_quad_approx_1000_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_044124.yml',
                              'sis_true_probs_myopic_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_045352.yml',
                              'sis_true_probs_myopic_quad_approx_100_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_030437.yml',
                              'sis_true_probs_myopic_quad_approx_50_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_044556.yml',
                              'sis_true_probs_myopic_quad_approx_50_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_025616.yml',
                              'sis_random_quad_approx_3000_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191126_235827.yml']
  fname_list_1_step_cutoff_more_reps = ['sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191127_050259.yml',
                                        'sis_random_quad_approx_100_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191127_031803.yml', 
                                        'sis_random_quad_approx_300_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191127_111855.yml']

  summarize_sampling_dbns(fname_list_1_step_cutoff, outname=None, save=False)
  summarize_sampling_dbns(fname_list_1_step_cutoff_more_reps, outname=None, save=False)




