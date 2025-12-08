#!/bin/bash

# Array of arguments
declare -a speed_tx=("20.0" "80.0")
declare -a speed_rx=("20.0" "80.0")
declare -a drop_idx=("6")
declare -a rx_ues_arr=("1" "2" "4" "6")

total_num_sims=$(( ${#speed_tx[@]} * ${#drop_idx[@]} * ${#rx_ues_arr[@]} ))

# Loop through the arrays
for speed_idx in ${!speed_tx[@]}; do
    for drop in ${!drop_idx[@]}; do
        for rx_ues_idx in ${!rx_ues_arr[@]}; do

            sim_num=$(( speed_idx * ${#drop_idx[@]} * ${#rx_ues_arr[@]} + drop * ${#rx_ues_arr[@]} + rx_ues_idx + 1 ))
            echo "Running simulation $sim_num out of $total_num_sims"

            python sims/sim_mu_mimo_minor_revision.py "${speed_tx[$speed_idx]}" "${drop_idx[$drop]}" "${rx_ues_arr[$rx_ues_idx]}"
        done
    done
done

#To make this file executable: chmod +x sims/run_sim_mu_mimo_minor_revision_multiple_drops.sh