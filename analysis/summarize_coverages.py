import yaml
import numpy
import pandas as pd
import argparse
import os
import pdb
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib as plt


def plot_coverages(df):
  sns.relplot(data=df, x='bandwidth', y='coverage', hue='L', style='beta', col='backup', kind='line')
  plt.show()
  return


def summarize_coverages_at_date(date_strs, save=False):
  results = {'L': [], 'beta': [], 'bandwidth': [], 'coverages': [], 'backup': []}
  any_found = False
  date_str_list = date_strs.split(',')
  for fname in os.listdir('coverages'):
    matches_date = False
    for date_str in date_str_list:
      if date_str in fname:
        matches_date = True
        break
    if matches_date:
      any_found = True
      full_fname = os.path.join('coverages', fname)
      f = yaml.load(open(full_fname, 'rb'))
      for bandwidth, bandwidth_results in f.items():
        results['L'].append(bandwidth_results['grid_size'])
        results['beta'].append(bandwidth_results['beta'])
        results['bandwidth'].append(bandwidth)
        results['coverages'].append(bandwidth_results['coverage'])
        results['backup'].append(bandwidth_results['backup'])
    df = pd.DataFrame.from_dict(results)
    df.sort_values(by=['backup', 'beta', 'L', 'bandwidth'], inplace=True)
    print(df)
    if save:
      df.to_csv(f'coverages/{date_strs}.csv')
  if not any_found:
    print('no results for that date.')

  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  args = parser.parse_args()
  summarize_coverages_at_date(args.date)
