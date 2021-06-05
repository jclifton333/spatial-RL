import numpy as np
import pandas as pd


if __name__ == "__main__":
  # Read csvs
  df1 = pd.read_csv('210518,210519,210520.csv')
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
  print(df)
