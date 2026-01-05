#!/bin/bash

set -euo pipefail

# Configuration
MOBILITY=${MOBILITY:-"high_mobility"}
declare -a RX_UES_ARR=("0" "2" "4" "6")
declare -a NUM_TXUE_SEL_ARR=("2" "4" "6" "8" "10")
MODULATION_ORDER=${MODULATION_ORDER:-4}
CODE_RATE=${CODE_RATE:-"1/2"}
PERFECT_CSI=${PERFECT_CSI:-False}
CSI_QUANTIZATION=${CSI_QUANTIZATION:-True}
LINK_ADAPT=${LINK_ADAPT:-True}
RL_MODE=${RL_MODE:-"deqn_plus_two_mode"} # "deqn", "deqn_plus_two_mode"

PARALLEL_JOBS=${PARALLEL_JOBS:-12}

TRAIN_DROP_START=${TRAIN_DROP_START:-4}
TRAIN_DROP_COUNT=${TRAIN_DROP_COUNT:-42}
TEST_DROP_START=${TEST_DROP_START:-1}
TEST_DROP_COUNT=${TEST_DROP_COUNT:-3}

TRAIN_END_DROP=$((TRAIN_DROP_START + TRAIN_DROP_COUNT - 1))
TRAIN_DROPS=$(seq -s, ${TRAIN_DROP_START} ${TRAIN_END_DROP})

TEST_END_DROP=$((TEST_DROP_START + TEST_DROP_COUNT - 1))

generate_scenario_args() {
    for RX_UES in "${RX_UES_ARR[@]}"; do
        for NUM_TXUE_SEL in "${NUM_TXUE_SEL_ARR[@]}"; do
            echo "${RX_UES} ${NUM_TXUE_SEL}"
        done
    done
}

run_scenario() {
    local RX_UES="$1"
    local NUM_TXUE_SEL="$2"
    local CHECKPOINT_DIR="results/rl_models/${MOBILITY}/drop_${TRAIN_END_DROP}_rx_UE_${RX_UES}_tx_UE_${NUM_TXUE_SEL}_imitation_none_steps_0"

    echo "Training ${RL_MODE} model for RX_UE=${RX_UES}, TX_UE=${NUM_TXUE_SEL} with ${TRAIN_DROP_COUNT} drops (${TRAIN_DROP_START}-${TRAIN_END_DROP}) in a single run"
    python sims/sim_mu_mimo_deqn_chan_pred_training.py \
        "${MOBILITY}" \
        "${TRAIN_DROPS}" \
        "${RX_UES}" \
        "${MODULATION_ORDER}" \
        "${CODE_RATE}" \
        "${NUM_TXUE_SEL}" \
        "${PERFECT_CSI}" \
        "${RL_MODE}" \
        "${CSI_QUANTIZATION}" \
        "${LINK_ADAPT}"

    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
        echo "Expected checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
        return 1
    fi

    echo "Testing with frozen model for drops ${TEST_DROP_START}-${TEST_END_DROP} (RX_UE=${RX_UES}, TX_UE=${NUM_TXUE_SEL})"
    for drop in $(seq ${TEST_DROP_START} ${TEST_END_DROP}); do
        echo "Running test drop ${drop} for RX_UE=${RX_UES}, TX_UE=${NUM_TXUE_SEL}"
        python sims/sim_mu_mimo_testing_updates.py \
            "${MOBILITY}" \
            "${drop}" \
            "${RX_UES}" \
            "${MODULATION_ORDER}" \
            "${CODE_RATE}" \
            "${NUM_TXUE_SEL}" \
            "${PERFECT_CSI}" \
            "${RL_MODE}" \
            "${CSI_QUANTIZATION}" \
            "${LINK_ADAPT}" \
            "${CHECKPOINT_DIR}" \
            "True"
            
    done
}

terminate_all_jobs() {
    local exit_code=${1:-1}

    echo "Error detected; terminating remaining jobs." >&2
    # Kill any still-running background jobs.
    for pid in $(jobs -p); do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null
        fi
    done

    # Ensure all children are reaped.
    wait
    exit "${exit_code}"
}

trap 'terminate_all_jobs $?' SIGINT SIGTERM

mapfile -t scenario_args < <(generate_scenario_args)

total_scenarios=${#scenario_args[@]}
running_jobs=0
completed_jobs=0
scenario_counter=0

for scenario in "${scenario_args[@]}"; do
    while (( running_jobs >= PARALLEL_JOBS )); do
        wait -n
        ((completed_jobs+=1))
        echo "Completed ${completed_jobs}/${total_scenarios} scenarios" >&2
        ((running_jobs-=1))
    done

    ((scenario_counter+=1))
    echo "Launching scenario ${scenario_counter}/${total_scenarios}: ${scenario}" >&2

    # shellcheck disable=SC2086
    run_scenario ${scenario} &
    ((running_jobs+=1))
done

while (( running_jobs > 0 )); do
    wait -n
    ((completed_jobs+=1))
    echo "Completed ${completed_jobs}/${total_scenarios} scenarios" >&2
    ((running_jobs-=1))
done

echo "Training and testing complete for all RX/TX UE combinations."