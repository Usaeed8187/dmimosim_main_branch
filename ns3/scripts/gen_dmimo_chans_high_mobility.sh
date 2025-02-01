#!/bin/bash

python main.py --seed 3007 --scenario V2V-Urban --small_scale_fading --num_subframes 100 --squad1_speed_km_h=10.0 --squad2_speed_km_h=10.0 --intra_sq1_rw_speed_km_h=1.0 --intra_sq2_rw_speed_km_h=1.0 --buildings_file 1narrow.txt  --squad1_direction=0 --squad2_direction=0.785398 --format=both

