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
                  'max_coverages': []}
  for fname in fname_list:
    d = yaml.load(open('./results/{}'.format(fname), 'rb'))
    res = d['results']
    
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
    # results_dict['nonzero_corr'].append(pearsonr(d['results']['mean_counts'], coverages)[0])
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
  big_ridge = ['sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191201_060443.yml',
               'sis_random_quad_approx_1000_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191201_035908.yml', 
               'sis_random_quad_approx_500_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191201_011039.yml', 
               'sis_true_probs_myopic_quad_approx_5000_nearestneighbor_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191201_093239.yml']

  small_features = ['sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191203_164743.yml',
'sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191203_020444.yml', 
'sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191203_133047.yml',
'sis_random_quad_approx_500_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191203_145034.yml',
'sis_random_quad_approx_2000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191203_215711.yml'
]

  ridge = ['sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191220_033702.yml',
           'sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191220_060631.yml',
           'sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191220_140037.yml',
           'sis_random_quad_approx_2000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191220_130940.yml']
  eigs = ['sis_random_quad_approx_100_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191221_235341.yml',
'sis_random_quad_approx_1000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191222_012809.yml',
'sis_random_quad_approx_2000_lattice_sampling-dbn-run=True_eval=two_step_mb_constant_cutoff_0.0_191222_060207.yml']


  simple_models_0_step = ['sis_random_quad_approx_50_lattice_eval-policy=one_step_eval_eval=one_step_eval_0.0_191228_172614.yml', 
'sis_random_quad_approx_100_lattice_eval-policy=one_step_eval_eval=one_step_eval_0.0_191228_172636.yml', 
'sis_random_quad_approx_500_lattice_eval-policy=one_step_eval_eval=one_step_eval_0.0_191228_172810.yml',
'sis_random_quad_approx_1000_lattice_eval-policy=one_step_eval_eval=one_step_eval_0.0_191228_173145.yml']
  simple_models = ['sis_random_quad_approx_500_lattice_eval-policy=two_step_mb_constant_cutoff_eval=two_step_mb_constant_cutoff_0.0_191228_165229.yml',
'sis_random_quad_approx_50_lattice_eval-policy=two_step_mb_constant_cutoff_eval=two_step_mb_constant_cutoff_0.0_191228_161827.yml',
'sis_random_quad_approx_2000_lattice_eval-policy=two_step_mb_constant_cutoff_eval=two_step_mb_constant_cutoff_0.0_191228_204424.yml', 
'sis_random_quad_approx_1000_lattice_eval-policy=two_step_mb_constant_cutoff_eval=two_step_mb_constant_cutoff_0.0_191228_214927.yml', 
'sis_random_quad_approx_100_lattice_eval-policy=two_step_mb_constant_cutoff_eval=two_step_mb_constant_cutoff_0.0_191228_162416.yml'] 
  one_step_binned = ['sis_random_quad_approx_50_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_012132.yml',
'sis_random_quad_approx_50_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_012132.yml',
'sis_random_quad_approx_100_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_012215.yml',
'sis_random_quad_approx_1000_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_012426.yml', 
'sis_treat_first_quad_approx_50_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_024142.yml',
'sis_treat_first_quad_approx_500_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_024459.yml']

  one_step_binned_more_reps = ['sis_random_quad_approx_50_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_021113.yml',
'sis_random_quad_approx_1000_lattice_eval-policy=one_step_bins_eval=one_step_bins_0.0_191229_021622.yml']
 
  one_step_parametric = ['sis_random_quad_approx_50_lattice_eval-policy=one_step_parametric_eval=one_step_parametric_0.0_200111_230907.yml',
			 'sis_random_quad_approx_100_lattice_eval-policy=one_step_parametric_eval=one_step_parametric_0.0_200111_230922.yml', 
 			 'sis_random_quad_approx_500_lattice_eval-policy=one_step_parametric_eval=one_step_parametric_0.0_200111_232241.yml', 
			 'sis_random_quad_approx_1000_lattice_eval-policy=one_step_parametric_eval=one_step_parametric_0.0_200111_232931.yml']
  one_step_parametric_true_model = ['sis_random_quad_approx_100_lattice_eval-policy=one_step_parametric_true_model_eval=one_step_parametric_true_model_0.0_200111_234750.yml']
  summarize_sampling_dbns(one_step_parametric_true_model, outname=None, save=False)
  # summarize_sampling_dbns(simple_models, outname=None, save=False)






