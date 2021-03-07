import os
import yaml
import argparse
import numpy as np


def summarize_results_at_date(date_str):
  epsilons = ['0.0', '0.5', '0.75', '1.0']
  networks = ['lattice', 'nearestneighbor']

  any_found = False
  for fname in os.listdir():
    if date_str in fname:
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
      if 'learn_embedding' in f['settings'].keys():
        to_print += ' {}'.format(f['settings']['learn_embedding'])

      print(to_print)
  if not any_found:
    print('No results found for that date.')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  args = parser.parse_args()
  summarize_results_at_date(args.date)
