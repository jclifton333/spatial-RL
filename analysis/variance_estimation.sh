#!/usr/bin/env bash

# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=0 --beta=0.1
# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=0 --beta=0.5
# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=0 --beta=1.0

# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=1 --beta=0.1
# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=1 --beta=0.5
# python3 variance_estimation.py --grid_size=3600 --n_rep=100 --backup=1 --beta=1.0

python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=0 --beta=0.1
python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=0 --beta=0.5
python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=0 --beta=1.0

# python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=1 --beta=0.1
# python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=1 --beta=0.5
# python3 variance_estimation.py --grid_size=4900 --n_rep=100 --backup=1 --beta=1.0

