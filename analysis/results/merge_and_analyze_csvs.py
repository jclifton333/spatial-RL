import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb


def merge_may_data():
  # Read csvs
  df1 = pd.read_csv('210514,210515,210516,210517,210518,210519,210520.csv')
  df2 = pd.read_csv('210531.csv')
  df3 = pd.read_csv('210601,210602.csv')

  # Add cols for raw features
  df1['raw0'] = 0
  df1['raw1'] = 0
  df2['raw0'] = 0
  df2['raw1'] = 1
  df3['raw0'] = 1
  df3['raw1'] = 1

  # Merge csvs
  df = pd.concat([df1, df2, df3])

  # Full env names
  df['full_env_name'] = None
  df.loc[df.env_name == 'Ebola', 'full_env_name'] = 'Ebola'
  df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'full_env_name'] = \
    'contrived' + df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'epsilon'].astype(str)
  df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'full_env_name'] = \
    df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'network'] + \
    df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'epsilon'].astype(str)

  # Full policy names
  df['full_policy_name'] = None
  df.loc[~df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'full_policy_name'] = \
    df.loc[~df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'policy_name']
  df.loc[df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'full_policy_name'] = \
    df.loc[df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'policy_name'] + \
    df.loc[df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'raw0'].astype(str) + \
    df.loc[df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'raw1'].astype(str)

  df.to_csv('0605_merged.csv')

  return


def barplots(df):
  full_env_names = ['lattice0.0', 'lattice0.5', 'lattice1.0']
  L_list = [100]
  df_subset = df[(df['full_env_name'].isin(full_env_names)) & (df['L'].isin(L_list))]
  sns.catplot(x='full_env_name', y='mean', hue='full_policy_name', kind='bar', data=df_subset)
  plt.show()
  return


if __name__ == "__main__":
  merge_may_data()
  df = pd.read_csv('0605_merged.csv')
  barplots(df)

