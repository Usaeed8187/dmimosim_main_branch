#!/bin/bash

# Sweep settings for sims/sim_mc_ncjt.py

declare -a drop_idx=($(seq 1 5))
declare -a rx_ues_arr=("3")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")
declare -a modulation_orders=("4" "6")
declare -a code_rates=("1/3" "1/2" "2/3")

PARALLEL_JOBS=${PARALLEL_JOBS:-12}

generate_args() {
    for j in "${!drop_idx[@]}"; do
        for k in "${!rx_ues_arr[@]}"; do
            for t in "${!num_txue_sel_arr[@]}"; do
                for m in "${!modulation_orders[@]}"; do
                    for c in "${!code_rates[@]}"; do
                        echo "Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, num_txue_sel: ${num_txue_sel_arr[$t]}, Modulation order: ${modulation_orders[$m]}, Code rate: ${code_rates[$c]}" >&2
                        echo "${drop_idx[$j]} ${rx_ues_arr[$k]} ${num_txue_sel_arr[$t]} ${modulation_orders[$m]} ${code_rates[$c]}"
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
    python sims/sim_mc_ncjt_single_cluster.py "${args[@]}"
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