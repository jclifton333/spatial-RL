import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb


def merge_ebola_data():
  df = pd.read_csv('210710,210711-ebola.csv')

  # Full env names
  df['full_env_name'] = None
  df.loc[df.env_name == 'Ebola', 'full_env_name'] = 'Ebola'

  # Full policy names
  df['full_policy_name'] = None
  df.loc[~df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'full_policy_name'] = \
    df.loc[~df.policy_name.isin(['one_step_ggcn', 'two_step_ggcn', 'two_step']), 'policy_name']

  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw0 == 1) & (df.raw1 == 1), 'full_policy_name'] = \
    'one_step_linear_raw'
  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw0 == 1) & (df.raw1 == 0), 'full_policy_name'] = \
    'one_step_linear'
  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw0 == 0) & (df.raw1 == 1), 'full_policy_name'] = \
    'one_step_ggcn_raw'
  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw0 == 0) & (df.raw1 == 0), 'full_policy_name'] = \
    'one_step_ggcn'

  df.loc[(df.policy_name == 'two_step') & (df.raw1 == 1), 'full_policy_name'] = \
    'two_step_linear_raw'
  df.loc[(df.policy_name == 'two_step') & (df.raw1 == 0), 'full_policy_name'] = \
    'two_step_linear'

  df.loc[(df.policy_name == 'two_step_ggcn') & (df.raw1 == 1), 'full_policy_name'] = \
    'two_step_ggcn_raw'
  df.loc[(df.policy_name == 'two_step_ggcn') & (df.raw1 == 0), 'full_policy_name'] = \
    'two_step_ggcn'

  # Overwrite with new tuned results
  df.loc[(df.policy_name == 'two_step_ggcn') & (df.raw1 == 0), 'mean'] = 0.160
  df.loc[(df.policy_name == 'two_step_ggcn') & (df.raw1 == 1), 'mean'] = 0.156
  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw1 == 0), 'mean'] = 0.150
  df.loc[(df.policy_name == 'one_step_ggcn') & (df.raw1 == 1), 'mean'] = 0.151

  df.to_csv('07_11_ebola_merged.csv')


def merge_may_data():
  # Read csvs
  df8 = pd.read_csv('210514,210515.csv')
  df0 = pd.read_csv('210516,210517.csv')
  df1 = pd.read_csv('210518,210519,210520.csv')
  df2 = pd.read_csv('210531.csv')
  df3 = pd.read_csv('210601,210602.csv')
  df4 = pd.read_csv('oracle-incorrect-contaminated.csv')
  df5 = pd.read_csv('210613,210614,210615,210616,210617.csv')
  df6 = pd.read_csv('210621,210622.csv')
  df7 = pd.read_csv('2108.csv')

  # Add cols for raw features
  df0.loc[df0['policy_name'] == 'one_step_ggcn', 'raw0'] = 0
  df0.loc[df0['policy_name'] == 'one_step_ggcn', 'raw1'] = 1
  df1['raw0'] = 0
  df1['raw1'] = 0
  df2['raw0'] = 0
  df2['raw1'] = 1
  df3['raw0'] = 1
  df3['raw1'] = 1
  df4['raw0'] = 0
  df4['raw1'] = 0
  df5['raw0'] = 0
  df5['raw1'] = 0
  df6['raw0'] = 0
  df6['raw1'] = 0

  # ToDo: not sure about this
  df8['raw0'] = 0
  df8['raw1'] = 0

  df4 = df4[~(df4['epsilon'].isin([0.5, 1.0]))]
  df5 = df5[df5['policy_name'] == 'two_step_true_probs']

  # Merge csvs
  df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8])

  # Full env names
  df['full_env_name'] = None
  df.loc[df.env_name == 'Ebola', 'full_env_name'] = 'Ebola'
  df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'full_env_name'] = \
    'contrived' + df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'epsilon'].astype(str) + \
    df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'L'].astype(str)
  df.loc[(df.env_name == 'sis') & (df.network.isnull()), 'network'] = 'contrived'
  df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'full_env_name'] = \
    df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'network'] + \
    df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'epsilon'].astype(str) + \
    df.loc[(df.env_name == 'sis') & (~df.network.isnull()), 'L'].astype(str)

    # Full policy names
  df['full_policy_name'] = None
  df.loc[~df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'full_policy_name'] = \
    df.loc[~df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'policy_name']
  df.loc[df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'full_policy_name'] = \
    df.loc[df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'policy_name'] + \
    df.loc[df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'raw0'].astype(str) + \
    df.loc[df.policy_name.isin(['one_step', 'one_step_ggcn', 'two_step_ggcn', 'two_step']), 'raw1'].astype(str)

  df.to_csv('0620_merged.csv')

  return


def barplots(df, normalize=False, ebola=False):
  # full_env_names = ['lattice0.0', 'lattice0.5', 'lattice1.0']
  if ebola:
    full_env_names = [name for name in df.full_env_name.unique()]
    L_list = df.L.unique()
    epsilon_list = df.epsilon.unique()
  else:
    full_env_names = [name for name in df.full_env_name.unique() if name != 'Ebola']
    L_list = [98, 100, 294, 300]
    epsilon_list = [0.0, 0.5, 1.0]
  # L_list = [98, 294]
  # L_list = df.L.unique()
  df_subset = df[(df['full_env_name'].isin(full_env_names)) & (df['L'].isin(L_list)) &
                 (df['epsilon'].isin(epsilon_list))]

  if normalize:
    df_subset['oracle_performance'] = None
    df_subset['random_performance'] = None
    env_to_flag_lst = []  # Flag envs whose best-performing policy is not in ['oracle ps', 'oracle two step']

    for full_env_name in df_subset.full_env_name.unique():
      df_subset_subset_policies = df_subset.loc[(df_subset['full_env_name'] == full_env_name),
                                       'policy_name'].to_list()


      # Take minimum of scores in case there are duplicates
      for policy_name_ in df_subset_subset_policies:
        df_subset.loc[(df_subset['full_env_name'] == full_env_name)
                       & (df_subset['full_policy_name'] == policy_name_), 'mean'] = \
          df_subset.loc[(df_subset['full_env_name'] == full_env_name)
                        & (df_subset['full_policy_name'] == policy_name_), 'mean'].min()

      # # Get oracle and random performances
      # if 'oracle_policy_search' in df_subset_subset_policies:
      #   oracle_ps_performance = \
      #     df_subset.loc[
      #                (df_subset['full_env_name'] == full_env_name)
      #               & (df_subset['full_policy_name'] == 'oracle_policy_search'), 'mean'].iloc[0]
      #   oracle_two_step_performance = \
      #     df_subset.loc[
      #                   (df_subset['full_env_name'] == full_env_name)
      #                   & (df_subset['full_policy_name'] == 'two_step_true_probs'), 'mean'].iloc[0]
      #   oracle_performance = np.min((oracle_ps_performance, oracle_two_step_performance))
      # else:
      #   oracle_performance = \
      #     df_subset.loc[
      #                   (df_subset['full_env_name'] == full_env_name)
      #                   & (df_subset['full_policy_name'] == 'two_step_true_probs'), 'mean'].iloc[0]
      # random_performance = \
      #   df_subset.loc[(df_subset['L'] == L)
      #             & (df_subset['full_env_name'] == full_env_name)
      #             & (df_subset['full_policy_name'] == 'random'), 'mean'].iloc[0]
      random_performance = df_subset.loc[
                   (df_subset['full_env_name'] == full_env_name), 'mean'].max()
      oracle_performance = df_subset.loc[
        (df_subset['full_env_name'] == full_env_name), 'mean'].min()
      df_subset.loc[df_subset.full_env_name == full_env_name, 'oracle_performance'] = \
        oracle_performance
      df_subset.loc[(df_subset.full_env_name == full_env_name), 'random_performance'] = \
        random_performance

      best_policy_name = df_subset.full_policy_name[df_subset.loc[
        (df_subset['full_env_name'] == full_env_name), 'mean'].idxmin()]
      if best_policy_name not in ['oracle_policy_search', 'two_step_true_probs']:
        df_subset.loc[df_subset['full_env_name'] == full_env_name, 'full_env_name'] += '*'

    # Normalize
    df_subset['normalized_mean'] = \
      (df_subset['mean'] - df_subset.oracle_performance) / (df_subset.random_performance - df_subset.oracle_performance)
    df_subset = df_subset[~df_subset['policy_name'].isin(['oracle_policy_search', 'random', 'two_step_true_probs'])]

    # Plot
    if ebola:
      policies_to_report = ['one_step_linear_raw', 'one_step_linear', 'one_step_ggcn',
                            'two_step_linear_raw', 'two_step_linear', 'two_step_ggcn',
                            'policy_search']
      remap_dict = {'one_step_ggcn': 'myopic_ggcn',
                    'two_step_ggcn': 'fqi_ggcn',
                    'one_step_linear': 'myopic_linear',
                    'two_step_linear': 'fqi_linear',
                    'two_step_linear_raw': 'fqi_linear_raw',
                    'one_step_linear_raw': 'myopic_linear_raw'}
      df_subset = df_subset[df_subset['full_policy_name'].isin(policies_to_report)]
      df_subset['full_policy_name'].replace(remap_dict, inplace=True)
      ci_width = (df_subset['upper'] - df_subset['lower']).max() / 2
      plot = sns.catplot(x='full_env_name', y='normalized_mean', hue='full_policy_name', kind='bar', data=df_subset,
                  ci=ci_width, legend=True)
      plot.savefig(f'ebola-results.png')
    else:
      df_subset.loc[(df_subset.env_name == 'sis') & (df_subset.network.isnull()), 'network'] = 'contrived'
      remap_dict = {'two_step_ggcn00': 'fqi_ggcn',
                    'one_step_ggcn00': 'myopic_linear',
                    'two_step01': 'fqi_linear',
                    'two_step11': 'fqi_linear_raw',
                    'one_step_ggcn01': 'myopic_linear_raw'}
      df_subset['full_policy_name'].replace(remap_dict, inplace=True)
      for env_name_ in ['lattice', 'nearestneighbor', 'contrived']:
        df_subset_subset = df_subset[df_subset['network'] == env_name_]
        plot = sns.catplot(x='epsilon', row='L',
                           y='normalized_mean', hue='full_policy_name', kind='bar',
                           legend=True, data=df_subset_subset)
        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(env_name_)
        plot.savefig(f'sis-{env_name_}-results.png')
  else:
    sns.catplot(x='full_env_name', y='mean', hue='full_policy_name', kind='bar', data=df_subset, col_wrap=3)
  # plt.show()
  return


if __name__ == "__main__":
  # merge_ebola_data()
  # df = pd.read_csv('07_11_ebola_merged.csv')
  merge_may_data()
  df = pd.read_csv('0620_merged.csv')
  barplots(df, normalize=True, ebola=False)
  # df4 = pd.read_csv('oracle-incorrect-contaminated.csv')

  # # Add cols for raw features
  # df4['raw0'] = 0
  # df4['raw1'] = 0

  # df4 = df4[~(df4['epsilon'].isin([0.5, 1.0]))]
  # print(df4)
