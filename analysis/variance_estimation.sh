#!/usr/bin/env bash

python3 variance_estimation.py --grid_size=100 --n_rep=100 --backup=0 --beta=1.0
python3 variance_estimation.py --grid_size=900 --n_rep=100 --backup=0 --beta=1.0
python3 variance_estimation.py --grid_size=2500 --n_rep=100 --backup=0 --beta=1.0

# python3 variance_estimation --grid_size=100 --n_rep=100 --backup=1 --beta=1.0
