#!/bin/bash

# Eval-only ChannelMamba runs across all default drops/mobilities using
# checkpoints produced by pooled training from twc_tput_across_tx_channelmamba.sh.

# Defaults aligned with sims/twc_tput_across_tx_channelmamba.sh
declare -a mobilities=("high_mobility" "higher_mobility" "highest_mobility")
declare -a drop_idx=($(seq 1 20))

declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a csi_quantization_arr=("True")

link_adapt="True"
CHANNELMAMBA_CHECKPOINT_ROOT=${CHANNELMAMBA_CHECKPOINT_ROOT:-results/channelmamba_checkpoints}
# Optional slot offset for eval-only runs. Default is 35.
CHANNELMAMBA_EVAL_START_SLOT_IDX=${CHANNELMAMBA_EVAL_START_SLOT_IDX:-35}
PARALLEL_JOBS=${PARALLEL_JOBS:-12}

run_scenario() {
    local args=("$@")
    python sims/sim_mu_mimo_testing_updates.py "${args[@]}"
}

mkdir -p "${CHANNELMAMBA_CHECKPOINT_ROOT}"

total_jobs=0
completed_jobs=0
failed_jobs=0
running_jobs=0

for mobility in "${mobilities[@]}"; do
    for d in "${drop_idx[@]}"; do
        for rx_ues in "${rx_ues_arr[@]}"; do
            for modulation_order in "${modulation_orders[@]}"; do
                for code_rate in "${code_rates[@]}"; do
                    for num_txue_sel in "${num_txue_sel_arr[@]}"; do
                        for perfect_csi in "${perfect_csi_arr[@]}"; do
                            for csi_quantization in "${csi_quantization_arr[@]}"; do
                                ((total_jobs++))
                                code_rate_sanitized="${code_rate//\//_}"
                                checkpoint_path="${CHANNELMAMBA_CHECKPOINT_ROOT}/cm_${mobility}_rx${rx_ues}_txsel${num_txue_sel}_m${modulation_order}_cr${code_rate_sanitized}_pcsi${perfect_csi}_q${csi_quantization}.pt"

                                echo "[eval-all-drops] job ${total_jobs}: mobility=${mobility}, drop=${d}, rx_ues=${rx_ues}, txsel=${num_txue_sel}, checkpoint_base=${checkpoint_path}" >&2
                                echo "[eval-all-drops] using start_slot_idx=${CHANNELMAMBA_EVAL_START_SLOT_IDX}" >&2

                                eval_args=(
                                    "${mobility}" "${d}" "${rx_ues}" "${modulation_order}" "${code_rate}" "${num_txue_sel}"
                                    "${perfect_csi}" "channelmamba" "${csi_quantization}" "${link_adapt}" "None" "True"
                                    "${checkpoint_path}" "eval" "" "${CHANNELMAMBA_EVAL_START_SLOT_IDX}"
                                )

                                run_scenario "${eval_args[@]}" &
                                ((running_jobs++))

                                while (( running_jobs >= PARALLEL_JOBS )); do
                                    if wait -n; then
                                        ((completed_jobs++))
                                        echo "[eval-all-drops] Completed ${completed_jobs}/${total_jobs}" >&2
                                    else
                                        ((failed_jobs++))
                                        echo "[eval-all-drops] A parallel eval job failed" >&2
                                    fi
                                    ((running_jobs--))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

while (( running_jobs > 0 )); do
    if wait -n; then
        ((completed_jobs++))
        echo "[eval-all-drops] Completed ${completed_jobs}/${total_jobs}" >&2
    else
        ((failed_jobs++))
        echo "[eval-all-drops] A parallel eval job failed" >&2
    fi
    ((running_jobs--))
done

echo "[eval-all-drops] finished: completed=${completed_jobs}, failed=${failed_jobs}, total=${total_jobs}" >&2