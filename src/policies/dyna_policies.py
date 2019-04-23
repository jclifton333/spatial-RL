def sis_one_step_dyna(**kwargs):
  env = kwargs['env'] 
  MAX_NUM_NONZERO = np.min((env.max_num_neighbors, 8))
  QUOTA = int(np.sqrt(env.L * env.T))

  bools = (0, 1)
  raw_feature_combos = [[i, j, k] for i in bools for j in bools for k in bools]

  # Build dictionary of feature indicator counts
  # ToDo: this shouldn't be done each time the policy is called...
  unique_feature_indicators = {}
  for n in range(MAX_NUM_NONZERO):
    truth_vals = [1 for i in range(n)[ + [0 for i in range(8 - n)]      
    for permutation_ in all_permutations(truth_vals):
      for raw_feature_combo in raw_feature_combos:
        feature_combo = tuple(raw_feature_combo + permutation_)
        unique_feature_indicators[feature_combo] = {'count': 0, 'list': []}

  # Count number of feature indicators
  X_indicator = np.vstack(env.X) > 0
  for x in X_indicator:
    unique_feature_indicators[tuple(x)]['count'] += 1  
    unique_feature_indicators[tuple(x)]['list'].append(x) 

  # Supplement features that fall short of quota
  X_synthetic = np.zeros((0, X_indicator.shape[1])) 
  for feature_info in unique_feature_indicators.values():
    count = feature_info['count']
    if count < QUOTA:
      num_fake_data = QUOTA - count 

      # Sample with replacement up to desired number


      
    


   

   
    

  

      

  
  

