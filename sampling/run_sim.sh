#!/bin/bash

python sample_trajs_machinek.py --row 0 --nsims 10 --save_id test --timestep 1e-7

python sample_trajs_gao.py --row 0 --nsims 5000 --save_id DNA23Metropolis_5000sims
