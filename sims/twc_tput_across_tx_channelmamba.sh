#!/bin/bash

# Array of arguments
# declare -a mobilities=("low_mobility" "medium_mobility" "high_mobility")
declare -a mobilities=("high_mobility" "higher_mobility" "highest_mobility")
# declare -a drop_idx=("26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "43" "44" "45")
# declare -a drop_idx=("1")
declare -a drop_idx=($(seq 1 20))
declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a csi_quantization_arr=("True")

link_adapt="True"

if [[ "${link_adapt}" == "True" ]]; then
    modulation_orders=("${modulation_orders[0]}")
    code_rates=("${code_rates[0]}")
fi

PARALLEL_JOBS=${PARALLEL_JOBS:-4}
CHANNELMAMBA_DROP_TRAIN_RATIO=${CHANNELMAMBA_DROP_TRAIN_RATIO:-0.5}
CHANNELMAMBA_CHECKPOINT_ROOT=${CHANNELMAMBA_CHECKPOINT_ROOT:-results/channelmamba_checkpoints}
failed_jobs=0

split_drops() {
    local total_drops=${#drop_idx[@]}

    if (( total_drops <= 1 )); then
        train_drop_count=${total_drops}
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
}

generate_setting_args() {
    for i in ${!mobilities[@]}; do
        for k in ${!rx_ues_arr[@]}; do
            for m in ${!modulation_orders[@]}; do
                for c in ${!code_rates[@]}; do
                    for t in ${!num_txue_sel_arr[@]}; do
                        for pcsi in ${!perfect_csi_arr[@]}; do
                            for cquant in ${!csi_quantization_arr[@]}; do
                                local csi_prediction_enabled="True"
                                local channel_prediction_method="channelmamba"

                                if [[ "${perfect_csi_arr[$pcsi]}" == "True" && "${csi_prediction_enabled}" == "True" ]]; then
                                    continue
                                fi
                                if [[ "${perfect_csi_arr[$pcsi]}" == "False" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                    continue
                                fi
                                if [[ "${csi_prediction_enabled}" == "True" && "${csi_quantization_arr[$cquant]}" == "False" ]]; then
                                    continue
                                fi

                                echo "Mobility: ${mobilities[$i]}, Rx UEs: ${rx_ues_arr[$k]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}, num_txue_sel: ${num_txue_sel_arr[$t]}, perfect_csi: ${perfect_csi_arr[$pcsi]}, channel_prediction_setting: channelmamba, csi_prediction: ${csi_prediction_enabled}, csi_quantization_on: ${csi_quantization_arr[$cquant]}, channel_prediction_method: ${channel_prediction_method}, link_adapt: ${link_adapt}" >&2
                                echo "${mobilities[$i]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi_arr[$pcsi]} ${csi_quantization_arr[$cquant]}"
                            done
                        done
                    done
                done
            done
        done
    done
}

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates.py "${args[@]}"
}

split_drops
echo "ChannelMamba drop split: train_drops=(${train_drops[*]}), test_drops=(${test_drops[*]}), ratio=${CHANNELMAMBA_DROP_TRAIN_RATIO}" >&2

mapfile -t setting_args < <(generate_setting_args)

total_settings=${#setting_args[@]}
setting_counter=0
completed_jobs=0

mkdir -p "${CHANNELMAMBA_CHECKPOINT_ROOT}"

for setting in "${setting_args[@]}"; do
    read -r mobility rx_ues modulation_order code_rate num_txue_sel perfect_csi csi_quantization <<< "${setting}"

    ((setting_counter++))
    echo "Launching setting ${setting_counter}/${total_settings}: mobility=${mobility}, rx_ues=${rx_ues}, num_txue_sel=${num_txue_sel}, perfect_csi=${perfect_csi}, csi_quantization=${csi_quantization}" >&2
    code_rate_sanitized="${code_rate//\//_}"
    checkpoint_path="${CHANNELMAMBA_CHECKPOINT_ROOT}/cm_${mobility}_rx${rx_ues}_txsel${num_txue_sel}_m${modulation_order}_cr${code_rate_sanitized}_pcsi${perfect_csi}_q${csi_quantization}.pt"

    checkpoint_root="${checkpoint_path%.pt}"
    rm -f "${checkpoint_root}"__tx*_rx*.pt

    train_drops_csv=$(IFS=,; echo "${train_drops[*]}")
    train_anchor_drop="${train_drops[0]}"
    echo "Launching pooled ChannelMamba train for setting ${setting_counter}/${total_settings} with train_drops=(${train_drops[*]})" >&2
    if ! run_scenario \
        "${mobility}" "${train_anchor_drop}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}" \
        "${perfect_csi}" "channelmamba" "${csi_quantization}" "${link_adapt}" "None" "True" \
        "${checkpoint_path}" "train" "${train_drops_csv}"; then
        ((failed_jobs++))
        echo "FAILED pooled ChannelMamba train for setting ${setting_counter}/${total_settings}" >&2
        continue
    fi
    ((completed_jobs++))
    echo "Completed ${completed_jobs} scenarios" >&2

    running_jobs=0
    for d in "${test_drops[@]}"; do
        while (( running_jobs >= PARALLEL_JOBS )); do
            if wait -n; then
                ((completed_jobs++))
                echo "Completed ${completed_jobs} scenarios" >&2
            else
                ((failed_jobs++))
                echo "A ChannelMamba eval scenario failed" >&2
            fi
            ((running_jobs--))
        done

        echo "Launching ChannelMamba eval drop=${d} for setting ${setting_counter}/${total_settings}" >&2
        run_scenario \
            "${mobility}" "${d}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}" \
            "${perfect_csi}" "channelmamba" "${csi_quantization}" "${link_adapt}" "None" "True" \
            "${checkpoint_path}" "eval" &
        ((running_jobs++))
    done

    while (( running_jobs > 0 )); do
        if wait -n; then
            ((completed_jobs++))
            echo "Completed ${completed_jobs} scenarios" >&2
        else
            ((failed_jobs++))
            echo "A ChannelMamba eval scenario failed" >&2
        fi
        ((running_jobs--))
    done
done

echo "All scenarios finished: completed=${completed_jobs}, failed=${failed_jobs}" >&2

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