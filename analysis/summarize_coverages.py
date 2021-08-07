import yaml
import numpy as np
import pandas as pd
import argparse
import os
import pdb
pd.set_option('display.max_rows', None)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_coverages(df):
  df['L'] = df['L'].astype('category')
  g = sns.relplot(data=df, x='bandwidth', y='coverages', hue='L', row='beta', col='backup', kind='line',
                  palette=['red', 'blue', 'orange'], legend=False)
  g.map(plt.axhline, y=0.95, c='black')
  custom_lines = [Line2D([0], [0], color='black', lw=1),
                  Line2D([0], [0], color='red', lw=1),
                  Line2D([0], [0], color='blue', lw=1),
                  Line2D([0], [0], color='orange', lw=1)]
  line_labels = ['Nominal coverage']
  L_sorted = np.array(df.L.unique())
  L_sorted.sort()
  for L_ in L_sorted:
    line_labels.append(f'L={L_}')
  # custom_lines = [Line2D([0], [0], color='black', lw=1)]
  plt.legend(custom_lines, line_labels, loc='lower right')
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
        # ToDo: throwing away data for grids bigger than 4000!
        if bandwidth_results['grid_size'] < 4000:
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
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--date', type=str)
  # args = parser.parse_args()
  # summarize_coverages_at_date(args.date)
  df = pd.read_csv('coverages/210713,210714,210715,210716,210717.csv')
  plot_coverages(df)

