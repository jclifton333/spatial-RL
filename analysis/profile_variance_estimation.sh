#!/usr/bin/env bash
python3 -m cProfile -o profile_output variance_estimation.py --n_rep=1 --grid_size=1600
python3 -m cprofilev -f profile_output
