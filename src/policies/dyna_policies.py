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

  mb_phats = [mb_clf(x) for x in env.X]

  # Count number of feature indicators
  for t, X_ in enumerate(env.X):
    for l, x in enumerate(X_):
      x_indicator = x > 0
      unique_feature_indicators[tuple(x_indicator)]['count'] += 1  
      unique_feature_indicators[tuple(x_indicator)]['list'].append((x, t, l)) 

  # Supplement features that fall short of quota
  X_synthetic = np.zeros((0, X_indicator.shape[1])) 
  Y_synthetic = np.zeros(0)
  for feature_info in unique_feature_indicators.values():
  count = feature_info['count']  # ToDo: what if count=0?
    if count < QUOTA:
      num_fake_data = QUOTA - count 

      # Sample with replacement up to desired number
      feature_list = feature_info['list']
      synthetic = np.random.choice(feature_list, num_fake_data, replace=T)
      x_synthetic = [o_[0] for o_ in feature_list]
      y_synthetic = [mb_phats[o_[1]][o_[2]] for o_ in feature_list]

      # Add to dataset
      X_synthetic = np.vstack((X_synthetic, x_synthetic))
      y_synthetic = np.hstack((Y_synthetic, y_synthetic))

   # Fit model-free model on new dataset
   X_new = np.vstack((np.vstack(env.X), X_synthetic)) 
   y_new = np.hstack((env.y, y_synthetic))


      
    


   

   
    

  

      

  
  

