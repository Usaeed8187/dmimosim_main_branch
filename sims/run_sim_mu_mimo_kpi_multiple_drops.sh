#!/bin/bash

# Array of arguments
# declare -a mobilities=("low_mobility" "medium_mobility" "high_mobility")
declare -a mobilities=("high_mobility")
# declare -a drop_idx=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
declare -a drop_idx=("1" "2" "3")
declare -a rx_ues_arr=("0" "2" "4" "6")
declare -a modulation_orders=("2" "4")
declare -a code_rates=("2/3" "5/6")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        for k in ${!rx_ues_arr[@]}; do
            for m in ${!modulation_orders[@]}; do
                for c in ${!code_rates[@]}; do
                    for t in ${!num_txue_sel_arr[@]}; do
                        echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}, num_txue_sel: ${num_txue_sel_arr[$t]}"
                        python sims/sim_mu_mimo_testing_updates.py "${mobilities[$i]}" "${drop_idx[$j]}" "${rx_ues_arr[$k]}" "${modulation_orders[$m]}" "${code_rates[$c]}" "${num_txue_sel_arr[$t]}"
                    done
                done
            done
        done
    done
done