#!/bin/bash

# Array of arguments
# declare -a mobilities=("low_mobility" "medium_mobility" "high_mobility")
declare -a mobilities=("low_mobility" "medium_mobility")
# declare -a drop_idx=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
declare -a drop_idx=("1" "2" "3" "4" "5" "7")
# declare -a rx_ues_arr=("1" "2" "4")
declare -a rx_ues_arr=("2")
declare -a precoding_method=("none")
declare -a receiver=("SIC")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        for k in ${!rx_ues_arr[@]}; do
        
            echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}"
            python sims/sim_ncjt_phase_3.py "${mobilities[$i]}" "${drop_idx[$j]}" "${precoding_method[$j]}" "${receiver[$j]}" "${rx_ues_arr[$k]}"
        done
    done
done