import os
import yaml
import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)


def summarize_results_at_date(date_strs):
  epsilons = ['0.0', '0.5', '0.75', '1.0']
  networks = ['lattice', 'nearestneighbor']
  results = {'env_name': [], 'network': [], 'policy_name': [], 'L': [], 'epsilon': [],
             'mean': [], 'se': []}
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
      se_ = np.std(scores) / np.sqrt(len(scores))
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
      results['policy_name'].append(policy_name)
      results['L'].append(L)
      results['epsilon'].append(epsilon)
      results['mean'].append(mean_)
      results['se'].append(se_)

      if 'learn_embedding' in f['settings'].keys():
        to_print += ' {}'.format(f['settings']['learn_embedding'])
      if 'raw_features' in f['settings'].keys():
        to_print += ' {}'.format(f['settings']['raw_features'])
      print(to_print)
  df = pd.DataFrame.from_dict(results)
  df.sort_values(by=['env_name', 'network', 'L', 'epsilon', 'policy_name'], inplace=True)
  print(df)
  if not any_found:
    print('No results found for that date.')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  args = parser.parse_args()
  summarize_results_at_date(args.date)
  
