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
  df4 = pd.read_csv('oracle-incorrect-contaminated.csv')

  # Add cols for raw features
  df1['raw0'] = 0
  df1['raw1'] = 0
  df2['raw0'] = 0
  df2['raw1'] = 1
  df3['raw0'] = 1
  df3['raw1'] = 1
  df4['raw0'] = 0
  df4['raw1'] = 0

  # Merge csvs
  df = pd.concat([df1, df2, df3, df4])

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


def barplots(df, normalize=False):
  # full_env_names = ['lattice0.0', 'lattice0.5', 'lattice1.0']
  # full_env_names = df.full_env_name.unique()
  full_env_names = [name for name in df.full_env_name.unique()
                    if (name != 'Ebola' and '0.5' not in name and '1.0' not in name)]  # ToDo: don't have oracle for Ebola
  L_list = [100]
  epsilon_list = [0.0]  # ToDo: currently only using epsilon=0.0 because oracles for epsilon > 0.0 are wrong
  df_subset = df[(df['full_env_name'].isin(full_env_names)) & (df['L'].isin(L_list)) &
                 (df['epsilon'].isin(epsilon_list))]

  if normalize:
    df_subset['oracle_performance'] = None
    df_subset['random_performance'] = None

    # Get oracle and random performances
    for L in L_list:
      for full_env_name in df_subset.full_env_name.unique():
        oracle_performance = \
          df_subset.loc[(df_subset['L'] == L)
                    & (df_subset['full_env_name'] == full_env_name)
                    & (df_subset['full_policy_name'] == 'oracle_policy_search'), 'mean'].iloc[0]
        random_performance = \
          df_subset.loc[(df_subset['L'] == L)
                    & (df_subset['full_env_name'] == full_env_name)
                    & (df_subset['full_policy_name'] == 'random'), 'mean'].iloc[0]
        df_subset.loc[(df_subset.L == L) & (df_subset.full_env_name == full_env_name), 'oracle_performance'] = \
          oracle_performance
        df_subset.loc[(df_subset.L == L) & (df_subset.full_env_name == full_env_name), 'random_performance'] = \
          random_performance

    # Normalize
    df_subset['normalized_mean'] = \
      (df_subset['mean'] - df_subset.oracle_performance) / (df_subset.random_performance - df_subset.oracle_performance)

    # Plot
    sns.catplot(x='full_env_name', y='normalized_mean', hue='full_policy_name', kind='bar', data=df_subset)
  else:
    sns.catplot(x='full_env_name', y='mean', hue='full_policy_name', kind='bar', data=df_subset)
  plt.show()
  return


if __name__ == "__main__":
  merge_may_data()
  df = pd.read_csv('0605_merged.csv')
  barplots(df, normalize=True)
