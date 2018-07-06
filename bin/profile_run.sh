#!/usr/bin/env bash
python -m cProfile -o profile_output ../src/run/run.py
cprofilev -f profile_output