import os
import yaml
import argparse


def summarize_results_at_date(date_str):
  any_found = False
  for fname in os.listdir():
    if date_str in fname:
      any_found = True
      f = yaml.load(open(fname, 'rb'))
      mean_, se_ = f['results']['mean'], f['results']['se']
      env_name, L, policy_name = f['settings']['env_name'], f['settings']['L'], f['settings']['policy_name']
      print(env_name, policy_name, L, mean_, se_)
  if not any_found:
    print('No results found for that date.')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--date', type=str)
  args = parser.parse_args()
  summarize_results_at_date(args.date)
