#!/usr/bin/env bash
python3 -m cProfile -o profile_output embedding.py
python3 -m cprofilev -f profile_output
