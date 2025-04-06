#!/bin/bash

nohup python sample_trajs_machinek.py --row 0 --nsims 300 --save_id test_0 > perfect_toehold7_reporter_test_0.log 2>&1 &

nohup python sample_trajs_machinek.py --row 1 --nsims 300 --save_id test_0 > proximal_toehold7_reporter_test_0.log 2>&1 &
