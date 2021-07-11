import os
import yaml
import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)


def bootstrap(x):
  N_REP = 10000
  n = len(x)
  x_mean = np.mean(x)
  diffs = []
  for rep in range(N_REP):
    x_boot = np.random.choice(x, size=n, replace=True)
    x_boot_mean = np.mean(x_boot)
    diffs.append(x_boot_mean-x_mean)
  lower_raw = np.percentile(diffs, 5)
  upper_raw = np.percentile(diffs, 95)
  lower = 2*x_mean - upper_raw
  upper = 2*x_mean - lower_raw
  se = np.sqrt(np.mean(np.array(diffs)**2))
  return se, lower, upper


def summarize_results_at_date(date_strs, save):
  epsilons = ['0.0', '0.5', '0.75', '1.0']
  networks = ['lattice', 'nearestneighbor']
  results = {'env_name': [], 'network': [], 'policy_name': [], 'L': [], 'epsilon': [],
             'mean': [], 'se': [], 'lower': [], 'upper': [], 'raw0': [], 'raw1': []}
  any_found = False 
  date_str_lst = date_strs.split(',')
  for fname in os.listdir():
    matches_date = False
    for date_str in date_str_lst: 
      if date_str in fname:
        matches_date = True
        break
    if matches_date:
      any_found = True
      f = yaml.load(open(fname, 'rb'))
      scores = [f['results'][ix]['score'] for ix in range(len(f['results'].keys()))]
      mean_ = np.mean(scores)
      # se_ = np.std(scores) / np.sqrt(len(scores))
      se_, lower_, upper_ = bootstrap(scores)
      env_name, L, policy_name = f['settings']['env_name'], f['settings']['L'], f['settings']['policy_name']

      epsilon = None
      for eps in epsilons:
        if eps in fname:
          epsilon = eps

      network = None
      for net in networks:
        if net in fname:
          network = net

      to_print = f'{env_name} {network} {policy_name} {L} {epsilon} {mean_} {se_}'
      results['env_name'].append(env_name)
      results['network'].append(network)
      results['L'].append(L)
      results['epsilon'].append(epsilon)
      results['mean'].append(mean_)
      results['se'].append(se_)
      results['lower'].append(lower_)
      results['upper'].append(upper_)

      if 'learn_embedding' in f['settings'].keys():
        to_print += ' {}'.format(f['settings']['learn_embedding'])
        if policy_name == 'one_step':
          learn_embedding = f['settings']['learn_embedding']
          if learn_embedding:
            policy_name += '_ggcn'
        results['raw0'].append(~f['settings']['learn_embedding'])
      if 'raw_features' in f['settings'].keys():
        to_print += ' {}'.format(f['settings']['raw_features'])
        results['raw1'].append(f['settings']['raw_features'])
      print(to_print)

      results['policy_name'].append(policy_name)

  df = pd.DataFrame.from_dict(results)
  df.sort_values(by=['env_name', 'network', 'L', 'epsilon', 'policy_name'], inplace=True)
  if save:
      df.to_csv(f'{date_strs}.csv')
  print(df)
  if not any_found:
    print('No results found for that date.')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  parser.add_argument('--save', type=str, default='False')
  args = parser.parse_args()
  save = (args.save == 'True')
  summarize_results_at_date(args.date, save)

  
