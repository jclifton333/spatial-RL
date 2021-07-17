import yaml
import numpy
import pandas as pd
import argparse


def summarize_coverages_at_date(date_strs):
  results = {'L': [], 'beta': [], 'bandwidth': [], 'coverages': [], 'backup': []}
  any_found = False
  date_str_list = date_strs.split(',')
  for fname in os.listdir():
    matches_date = False
    for date_str in date_str_list:
      if date_str in fname:
        matches_date = True
        break
    if matches_date:
      any_found = True
      f = yaml.load(open(fname, 'rb'))
      for bandwidth, bandwidth_results in f.items():
        results['L'].append(bandwidth_results['grid_size'])
        results['beta'].append(bandwidth_results['beta'])
        results['bandwidth'].append(bandwidth)
        results['coverages'].append(bandwidth_results['coverage'])
        results['backup'].append(bandwidth_results['backup'])
    df = pd.DataFrame.from_dict(results)
    df.sort_values(by=['backup', 'L', 'beta', 'bandwidth'])
  if not any_found:
    print('no results for that date.')

  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  args = parser.parse_args()
  summarize_coverages_at_date(args.date)
