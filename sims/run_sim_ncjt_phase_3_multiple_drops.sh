#!/bin/bash

# Array of arguments
declare -a mobilities=("high_mobility" "medium_mobility" "low_mobility" )
# declare -a mobilities=("high_mobility")
# declare -a drop_idx=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
declare -a drop_idx=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# declare -a rx_ues_arr=("1" "2" "4")
declare -a rx_ues_arr=("2" "4")
# declare -a precoding_method=("none" "eigenmode")
declare -a precoding_method=("eigenmode")
declare -a receiver=("SIC")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        for k in ${!rx_ues_arr[@]}; do
            for l in ${!precoding_method[@]}; do
                for m in ${!receiver[@]}; do    
        
                    echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, 
                        Rx UEs: ${rx_ues_arr[$k]}, precoding_method: ${precoding_method[$l]}, receiver: ${receiver[$m]}"

                    python sims/sim_ncjt_phase_3.py "${mobilities[$i]}" "${drop_idx[$j]}" "${precoding_method[$l]}" "${receiver[$m]}" "${rx_ues_arr[$k]}"
                done
            done
        done
    done
done