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
declare -a channel_prediction_settings=("None" "weiner_filter" "two_mode") # "None", "weiner_filter", "two_mode". If "None", cfg.csi_prediction = False. otherwise, cfg.csi_prediction = True and cfg.channel_prediction_method is changed accordingly.
declare -a csi_quantization_arr=("True" "False")

link_adapt="True"

if [[ "${link_adapt}" == "True" ]]; then
    modulation_orders=("${modulation_orders[0]}")
    code_rates=("${code_rates[0]}")
fi

PARALLEL_JOBS=${PARALLEL_JOBS:-4}
PARALLEL_EXTRA_OPTS=${PARALLEL_EXTRA_OPTS:-}


generate_args() {
    # Loop through the arrays
    for i in ${!mobilities[@]}; do
        for j in ${!drop_idx[@]}; do
            for k in ${!rx_ues_arr[@]}; do
                for m in ${!modulation_orders[@]}; do
                    for c in ${!code_rates[@]}; do
                        for t in ${!num_txue_sel_arr[@]}; do
                            for pcsi in ${!perfect_csi_arr[@]}; do
                                for cp_setting in ${!channel_prediction_settings[@]}; do
                                    for cquant in ${!csi_quantization_arr[@]}; do
                                        channel_prediction_setting=${channel_prediction_settings[$cp_setting]}
                                        csi_prediction_enabled="False"
                                        channel_prediction_method="None"
                                        if [[ "${channel_prediction_setting}" != "None" ]]; then
                                            csi_prediction_enabled="True"
                                            channel_prediction_method=${channel_prediction_setting}
                                        fi

                                        if [[ "${perfect_csi_arr[$pcsi]}" == "True" && "${csi_prediction_enabled}" == "True" ]]; then
                                            continue
                                        fi
                                        if [[ "${perfect_csi_arr[$pcsi]}" == "False" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                            continue
                                        fi
                                        if [[ "${csi_prediction_enabled}" == "True" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                            continue
                                        fi

                                        echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}, num_txue_sel: ${num_txue_sel_arr[$t]}, perfect_csi: ${perfect_csi_arr[$pcsi]}, channel_prediction_setting: ${channel_prediction_setting}, csi_prediction: ${csi_prediction_enabled}, csi_quantization_on: ${csi_quantization_arr[$cquant]}, channel_prediction_method: ${channel_prediction_method}, link_adapt: ${link_adapt}" >&2
                                        echo "${mobilities[$i]} ${drop_idx[$j]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi_arr[$pcsi]} ${channel_prediction_setting} ${csi_quantization_arr[$cquant]} ${link_adapt}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
}

generate_args | parallel -j "${PARALLEL_JOBS}" ${PARALLEL_EXTRA_OPTS} --colsep ' ' python sims/sim_mu_mimo_testing_updates.py {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}

# Reference table
# Perfect CSI |  Prediction | Quantization | Meaning
#------------------------------------------------------
#     F       |      F      |     F        | Not simulated
#     F       |      F      |     T        | Worst case: imperfect channel estimation, quantized CSI feedback without prediction
#     F       |      T      |     F        | Not simulated
#     F       |      T      |     T        | Achievable case: imperfect channel estimation, CSI prediction, quantized CSI feedback
#     T       |      F      |     F        | Ideal case: perfect CSI at the BS (perfect channel estimation, no delay, no quantization)
#     T       |      F      |     T        | Semi-ideal case: perfect CSI at the UE, quantized CSI feedback
#     T       |      T      |     F        | Not simulated
#     T       |      T      |     T        | Not simulated