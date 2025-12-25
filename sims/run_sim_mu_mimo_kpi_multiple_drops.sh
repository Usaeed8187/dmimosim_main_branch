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
declare -a perfect_csi_arr=("True" "False")
declare -a csi_prediction_arr=("False")
declare -a csi_quantization_arr=("True" "False")

# Loop through the arrays
for i in ${!mobilities[@]}; do
    for j in ${!drop_idx[@]}; do
        for k in ${!rx_ues_arr[@]}; do
            for m in ${!modulation_orders[@]}; do
                for c in ${!code_rates[@]}; do
                    for t in ${!num_txue_sel_arr[@]}; do
                        for pcsi in ${!perfect_csi_arr[@]}; do
                            for cpred in ${!csi_prediction_arr[@]}; do
                                for cquant in ${!csi_quantization_arr[@]}; do
                                    if [[ "${perfect_csi_arr[$pcsi]}" == "True" && "${csi_prediction_arr[$cpred]}" == "True" ]]; then
                                        continue
                                    fi
                                    if [[ "${perfect_csi_arr[$pcsi]}" == "False" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                        continue
                                    fi
                                    if [[ "${csi_prediction_arr[$cpred]}" == "True" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                        continue
                                    fi
                                    echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}, num_txue_sel: ${num_txue_sel_arr[$t]}, perfect_csi: ${perfect_csi_arr[$pcsi]}, csi_prediction: ${csi_prediction_arr[$cpred]}, csi_quantization_on: ${csi_quantization_arr[$cquant]}"
                                    python sims/sim_mu_mimo_testing_updates.py "${mobilities[$i]}" "${drop_idx[$j]}" "${rx_ues_arr[$k]}" "${modulation_orders[$m]}" "${code_rates[$c]}" "${num_txue_sel_arr[$t]}" "${perfect_csi_arr[$pcsi]}" "${csi_prediction_arr[$cpred]}" "${csi_quantization_arr[$cquant]}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


# Reference table
# Perfect CSI |  Prediction | Quantization | Meaning
#------------------------------------------------------
#     F       |      F      |     F        | Not simulated
#     F       |      F      |     T        | Baseline: imperfect channel estimation, quantized CSI feedback without prediction
#     F       |      T      |     F        | Not simulated
#     F       |      T      |     T        | Achievable case: imperfect channel estimation, CSI prediction, quantized CSI feedback
#     T       |      F      |     F        | Ideal case: perfect CSI at the BS (perfect channel estimation, no delay, no quantization)
#     T       |      F      |     T        | Semi-ideal case: perfect CSI at the UE, quantized CSI feedback
#     T       |      T      |     F        | Not simulated
#     T       |      T      |     T        | Not simulated