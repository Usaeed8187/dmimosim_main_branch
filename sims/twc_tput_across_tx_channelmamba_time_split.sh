#!/bin/bash

# ChannelMamba time-split experiment:
# - Offline train on first half of time samples pooled across ALL drops.
# - Eval on second half of time samples across ALL drops.

declare -a mobilities=("high_mobility" "higher_mobility" "highest_mobility")
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

PARALLEL_JOBS=${PARALLEL_JOBS:-12}
CHANNELMAMBA_TIME_TRAIN_RATIO=${CHANNELMAMBA_TIME_TRAIN_RATIO:-0.7}
CHANNELMAMBA_CHECKPOINT_ROOT=${CHANNELMAMBA_CHECKPOINT_ROOT:-results/channelmamba_checkpoints}
CHANNELMAMBA_TRAIN_START_SLOT_IDX=${CHANNELMAMBA_TRAIN_START_SLOT_IDX:-33}
CHANNELMAMBA_TOTAL_SLOTS=${CHANNELMAMBA_TOTAL_SLOTS:-100}
CHANNELMAMBA_CSI_DELAY=${CHANNELMAMBA_CSI_DELAY:-4}
failed_jobs=0

CHANNELMAMBA_EVAL_START_SLOT_IDX=${CHANNELMAMBA_EVAL_START_SLOT_IDX:-$(python - <<PY
import math
start = int("${CHANNELMAMBA_TRAIN_START_SLOT_IDX}")
total_slots = int("${CHANNELMAMBA_TOTAL_SLOTS}")
csi_delay = int("${CHANNELMAMBA_CSI_DELAY}")
ratio = float("${CHANNELMAMBA_TIME_TRAIN_RATIO}")
if csi_delay <= 0:
    raise SystemExit("CHANNELMAMBA_CSI_DELAY must be > 0")
slot_indices = list(range(start, total_slots, csi_delay))
if not slot_indices:
    raise SystemExit("No slot indices produced; check train start / total slots / csi delay")
train_count = int(math.floor(len(slot_indices) * ratio))
train_count = max(1, min(train_count, len(slot_indices)))
# eval begins at first slot after the training prefix; if ratio==1.0, reuse last slot
idx = min(train_count, len(slot_indices) - 1)
print(slot_indices[idx])
PY
)}

all_drops_csv=$(IFS=,; echo "${drop_idx[*]}")

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates.py "${args[@]}"
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

                                echo "${mobilities[$i]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi_arr[$pcsi]} ${csi_quantization_arr[$cquant]}"
                            done
                        done
                    done
                done
            done
        done
    done
}

mkdir -p "${CHANNELMAMBA_CHECKPOINT_ROOT}"
mapfile -t setting_args < <(generate_setting_args)

total_settings=${#setting_args[@]}
setting_counter=0
completed_jobs=0

echo "ChannelMamba time split: train_time_ratio=${CHANNELMAMBA_TIME_TRAIN_RATIO}, train_start_slot_idx=${CHANNELMAMBA_TRAIN_START_SLOT_IDX}, eval_start_slot_idx=${CHANNELMAMBA_EVAL_START_SLOT_IDX}, total_slots=${CHANNELMAMBA_TOTAL_SLOTS}, csi_delay=${CHANNELMAMBA_CSI_DELAY}, pooled_train_drops=(${drop_idx[*]})" >&2

for setting in "${setting_args[@]}"; do
    read -r mobility rx_ues modulation_order code_rate num_txue_sel perfect_csi csi_quantization <<< "${setting}"

    ((setting_counter++))
    code_rate_sanitized="${code_rate//\//_}"
    checkpoint_path="${CHANNELMAMBA_CHECKPOINT_ROOT}/cm_timesplit_${mobility}_rx${rx_ues}_txsel${num_txue_sel}_m${modulation_order}_cr${code_rate_sanitized}_pcsi${perfect_csi}_q${csi_quantization}.pt"

    checkpoint_root="${checkpoint_path%.pt}"
    rm -f "${checkpoint_root}"__tx*_rx*.pt

    train_anchor_drop="${drop_idx[0]}"
    echo "Launching pooled ChannelMamba TIME-SPLIT train for setting ${setting_counter}/${total_settings}" >&2
    if ! run_scenario \
        "${mobility}" "${train_anchor_drop}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}" \
        "${perfect_csi}" "channelmamba" "${csi_quantization}" "${link_adapt}" "None" "True" \
        "${checkpoint_path}" "train" "${all_drops_csv}" "${CHANNELMAMBA_TRAIN_START_SLOT_IDX}" "within_drop" "" "time_split" "${CHANNELMAMBA_TIME_TRAIN_RATIO}"; then
        ((failed_jobs++))
        echo "FAILED pooled ChannelMamba time-split train for setting ${setting_counter}/${total_settings}" >&2
        continue
    fi
    ((completed_jobs++))

    running_jobs=0
    for d in "${drop_idx[@]}"; do
        while (( running_jobs >= PARALLEL_JOBS )); do
            if wait -n; then
                ((completed_jobs++))
            else
                ((failed_jobs++))
            fi
            ((running_jobs--))
        done

        echo "Launching ChannelMamba TIME-SPLIT eval drop=${d} for setting ${setting_counter}/${total_settings}" >&2
        run_scenario \
            "${mobility}" "${d}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}" \
            "${perfect_csi}" "channelmamba" "${csi_quantization}" "${link_adapt}" "None" "True" \
            "${checkpoint_path}" "eval" "" "${CHANNELMAMBA_EVAL_START_SLOT_IDX}" "within_drop" "" "time_split" "${CHANNELMAMBA_TIME_TRAIN_RATIO}" &
        ((running_jobs++))
    done

    while (( running_jobs > 0 )); do
        if wait -n; then
            ((completed_jobs++))
        else
            ((failed_jobs++))
        fi
        ((running_jobs--))
    done
done

echo "ChannelMamba time-split scenarios finished: completed=${completed_jobs}, failed=${failed_jobs}" >&2