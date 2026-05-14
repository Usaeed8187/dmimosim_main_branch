#!/bin/bash

declare -a mobilities=("high_mobility" "higher_mobility" "highest_mobility")
declare -a drop_idx=($(seq 1 20))
declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a csi_quantization_arr=("True")

link_adapt="True"
WESN_DROP_TRAIN_RATIO=${WESN_DROP_TRAIN_RATIO:-0.5}
PARALLEL_JOBS=${PARALLEL_JOBS:-8}
failed_jobs=0

split_drops() {
    local total_drops=${#drop_idx[@]}
    if (( total_drops <= 1 )); then
        train_drop_count=${total_drops}
    else
        train_drop_count=$(python - <<PY
import math
n = ${#drop_idx[@]}
ratio = float("${WESN_DROP_TRAIN_RATIO}")
count = int(math.floor(n * ratio))
count = max(1, min(count, n - 1))
print(count)
PY
)
    fi
    train_drops=("${drop_idx[@]:0:${train_drop_count}}")
    test_drops=("${drop_idx[@]:${train_drop_count}}")
}

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates.py "${args[@]}"
}

generate_setting_args() {
    for mobility in "${mobilities[@]}"; do
        for rx_ues in "${rx_ues_arr[@]}"; do
            for modulation_order in "${modulation_orders[@]}"; do
                for code_rate in "${code_rates[@]}"; do
                    for num_txue_sel in "${num_txue_sel_arr[@]}"; do
                        for perfect_csi in "${perfect_csi_arr[@]}"; do
                            for csi_quantization in "${csi_quantization_arr[@]}"; do
                                echo "${mobility} ${rx_ues} ${modulation_order} ${code_rate} ${num_txue_sel} ${perfect_csi} ${csi_quantization}"
                            done
                        done
                    done
                done
            done
        done
    done
}

split_drops
train_drops_csv=$(IFS=,; echo "${train_drops[*]}")
echo "Configured WESN drop split: train_drops=(${train_drops[*]}), test_drops=(${test_drops[*]}), ratio=${WESN_DROP_TRAIN_RATIO}" >&2

mapfile -t setting_args < <(generate_setting_args)

for setting in "${setting_args[@]}"; do
    read -r mobility rx_ues modulation_order code_rate num_txue_sel perfect_csi csi_quantization <<< "${setting}"

    running_jobs=0
    for d in "${test_drops[@]}"; do
        while (( running_jobs >= PARALLEL_JOBS )); do
            if ! wait -n; then
                ((failed_jobs++))
            fi
            ((running_jobs--))
        done

        run_scenario \
            "${mobility}" "${d}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}" \
            "${perfect_csi}" "configured_wesn" "${csi_quantization}" "${link_adapt}" "None" "True" \
            "" "train" "" "33" "across_drops" "${train_drops_csv}" &
        ((running_jobs++))
    done

    while (( running_jobs > 0 )); do
        if ! wait -n; then
            ((failed_jobs++))
        fi
        ((running_jobs--))
    done
done

echo "Configured WESN drop-split finished. failed=${failed_jobs}" >&2