#!/bin/bash

# nohup python sample_trajs_machinek.py --row 0 --nsims 5000 --save_id 1e-7 --timestep 1e-7 > perfect_toehold7_reporter_1e-5.log 2>&1 &

python sample_trajs_machinek.py --row 0 --nsims 10 --save_id 1e-7 --timestep 1e-7


# nohup python sample_trajs_machinek.py --row 1 --nsims 2000 --save_id test_0 > proximal_toehold7_reporter_test_0.log 2>&1 &