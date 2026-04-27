#!/bin/bash

# Array of arguments
# declare -a mobilities=("low_mobility" "medium_mobility" "high_mobility")
declare -a mobilities=("highest_mobility")
# declare -a drop_idx=("26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "43" "44" "45")
# declare -a drop_idx=("1")
declare -a drop_idx=($(seq 6 10))
declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("8")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a channel_prediction_settings=("kalman_filter") # "None" "weiner_filter" "two_mode". If "None", cfg.csi_prediction = False. otherwise, cfg.csi_prediction = True and cfg.channel_prediction_method is changed accordingly.
declare -a csi_quantization_arr=("True")

link_adapt="True"

if [[ "${link_adapt}" == "True" ]]; then
    modulation_orders=("${modulation_orders[0]}")
    code_rates=("${code_rates[0]}")
fi

PARALLEL_JOBS=${PARALLEL_JOBS:-12}
CHANNELMAMBA_DROP_TRAIN_RATIO=${CHANNELMAMBA_DROP_TRAIN_RATIO:-0.5}
CHANNELMAMBA_CHECKPOINT_ROOT=${CHANNELMAMBA_CHECKPOINT_ROOT:-results/channelmamba_checkpoints}

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
    done
}

mapfile -t scenario_args < <(generate_args)

total_scenarios=${#scenario_args[@]}
running_jobs=0
completed_jobs=0
scenario_counter=0

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates.py "${args[@]}"
}

if [[ ${#channel_prediction_settings[@]} -eq 1 && "${channel_prediction_settings[0]}" == "channelmamba" ]]; then
    if [[ ${#drop_idx[@]} -le 1 ]]; then
        train_drop_count=${#drop_idx[@]}
    else
        train_drop_count=$(python - <<PY
import math
n = ${#drop_idx[@]}
ratio = float("${CHANNELMAMBA_DROP_TRAIN_RATIO}")
count = int(math.floor(n * ratio))
count = max(1, min(count, n - 1))
print(count)
PY
)
    fi

    train_drops=("${drop_idx[@]:0:${train_drop_count}}")
    test_drops=("${drop_idx[@]:${train_drop_count}}")

    echo "ChannelMamba drop split: train_drops=(${train_drops[*]}), test_drops=(${test_drops[*]}), ratio=${CHANNELMAMBA_DROP_TRAIN_RATIO}" >&2

    mkdir -p "${CHANNELMAMBA_CHECKPOINT_ROOT}"
    checkpoint_path="${CHANNELMAMBA_CHECKPOINT_ROOT}/channelmamba_mob_${mobilities[0]}_tx_${num_txue_sel_arr[0]}_rx_${rx_ues_arr[0]}.pt"
    rm -f "${checkpoint_path}"

    # Train stage (sequential): progressively fit on first N% drops and overwrite checkpoint each run.
    for d in "${train_drops[@]}"; do
        ((scenario_counter++))
        echo "Launching ChannelMamba train scenario ${scenario_counter} drop=${d}" >&2
        run_scenario \
            "${mobilities[0]}" "${d}" "${rx_ues_arr[0]}" "${modulation_orders[0]}" "${code_rates[0]}" "${num_txue_sel_arr[0]}" \
            "${perfect_csi_arr[0]}" "channelmamba" "${csi_quantization_arr[0]}" "${link_adapt}" "None" "True" \
            "${checkpoint_path}" "train"
        ((completed_jobs++))
        echo "Completed ${completed_jobs} scenarios" >&2
    done

    # Eval stage (parallel): load frozen checkpoint for remaining drops.
    for d in "${test_drops[@]}"; do
        while (( running_jobs >= PARALLEL_JOBS )); do
            wait -n
            ((completed_jobs++))
            echo "Completed ${completed_jobs} scenarios" >&2
            ((running_jobs--))
        done

        ((scenario_counter++))
        echo "Launching ChannelMamba eval scenario ${scenario_counter} drop=${d}" >&2
        run_scenario \
            "${mobilities[0]}" "${d}" "${rx_ues_arr[0]}" "${modulation_orders[0]}" "${code_rates[0]}" "${num_txue_sel_arr[0]}" \
            "${perfect_csi_arr[0]}" "channelmamba" "${csi_quantization_arr[0]}" "${link_adapt}" "None" "True" \
            "${checkpoint_path}" "eval" &
        ((running_jobs++))
    done

    while (( running_jobs > 0 )); do
        wait -n
        ((completed_jobs++))
        echo "Completed ${completed_jobs} scenarios" >&2
        ((running_jobs--))
    done
else
    for scenario in "${scenario_args[@]}"; do
        # Throttle concurrency to PARALLEL_JOBS
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

    # Wait for any remaining background jobs
    while (( running_jobs > 0 )); do
        wait -n
        ((completed_jobs++))
        echo "Completed ${completed_jobs}/${total_scenarios} scenarios" >&2
        ((running_jobs--))
    done
fi

echo "All ${completed_jobs} scenarios completed" >&2

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