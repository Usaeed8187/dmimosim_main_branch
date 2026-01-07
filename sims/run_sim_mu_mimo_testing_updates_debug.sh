#!/bin/bash

declare -a mobilities=("high_mobility")
declare -a drop_idx=({1..30})
declare -a rx_ues_arr=("4")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a num_txue_sel_arr=("4")

link_adapt="True"
channel_prediction_setting="None"
csi_quantization_on="True"
rl_checkpoint=""
rl_evaluation_only="False"

declare -a scenarios=(
    "perfect_w1 False W_1 False"
    "perfect_wc1 False W_C1 False"
    "perfect_wc2_t False W_C2_t False"
    "perfect_everything True None False"
    "outdated_everything False None True"
)

if [[ "${link_adapt}" == "True" ]]; then
    modulation_orders=("${modulation_orders[0]}")
    code_rates=("${code_rates[0]}")
fi

PARALLEL_JOBS=${PARALLEL_JOBS:-12}

generate_args() {
    for i in ${!mobilities[@]}; do
        for j in ${!drop_idx[@]}; do
            for k in ${!rx_ues_arr[@]}; do
                for m in ${!modulation_orders[@]}; do
                    for c in ${!code_rates[@]}; do
                        for t in ${!num_txue_sel_arr[@]}; do
                            for scenario in "${scenarios[@]}"; do
                                read -r scenario_name perfect_csi perfect_csi_matrix force_outdated_csi <<< "${scenario}"

                                echo "Scenario: ${scenario_name}, Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}, num_txue_sel: ${num_txue_sel_arr[$t]}, perfect_csi: ${perfect_csi}, perfect_csi_matrix: ${perfect_csi_matrix}, force_outdated_csi: ${force_outdated_csi}, channel_prediction_setting: ${channel_prediction_setting}, csi_quantization_on: ${csi_quantization_on}, link_adapt: ${link_adapt}" >&2

                                echo "${mobilities[$i]} ${drop_idx[$j]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi} ${channel_prediction_setting} ${csi_quantization_on} ${link_adapt} ${rl_checkpoint} ${rl_evaluation_only} ${perfect_csi_matrix} ${force_outdated_csi}"
                            done
                        done
                    done
                done
            done
        done
    done
}

mapfile -t scenario_args < <(generate_args)

total_scenarios=${#scenario_args[@]}
running_jobs=0
completed_jobs=0
scenario_counter=0

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates_debug.py "${args[@]}"
}

for scenario in "${scenario_args[@]}"; do
    while (( running_jobs >= PARALLEL_JOBS )); do
        wait -n
        ((completed_jobs++))
        echo "Completed ${completed_jobs}/${total_scenarios} scenarios" >&2
        ((running_jobs--))
    done

    ((scenario_counter++))
    echo "Launching scenario ${scenario_counter}/${total_scenarios}" >&2

    # shellcheck disable=SC2086
    run_scenario ${scenario} &
    ((running_jobs++))
done

while (( running_jobs > 0 )); do
    wait -n
    ((completed_jobs++))
    echo "Completed ${completed_jobs}/${total_scenarios} scenarios" >&2
    ((running_jobs--))
done

echo "All ${completed_jobs}/${total_scenarios} scenarios completed" >&2