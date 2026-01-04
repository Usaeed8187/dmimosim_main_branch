#!/bin/bash

set -euo pipefail

# Configuration
MOBILITY=${MOBILITY:-"high_mobility"}
RX_UES=${RX_UES:-2}
MODULATION_ORDER=${MODULATION_ORDER:-4}
CODE_RATE=${CODE_RATE:-"1/2"}
NUM_TXUE_SEL=${NUM_TXUE_SEL:-2}
PERFECT_CSI=${PERFECT_CSI:-False}
CSI_QUANTIZATION=${CSI_QUANTIZATION:-True}
LINK_ADAPT=${LINK_ADAPT:-True}

TRAIN_DROP_START=${TRAIN_DROP_START:-1}
TRAIN_DROP_COUNT=${TRAIN_DROP_COUNT:-3}
TEST_DROP_START=${TEST_DROP_START:-$((TRAIN_DROP_START + TRAIN_DROP_COUNT))}
TEST_DROP_COUNT=${TEST_DROP_COUNT:-2}

TRAIN_END_DROP=$((TRAIN_DROP_START + TRAIN_DROP_COUNT - 1))
TRAIN_DROPS=$(seq -s, ${TRAIN_DROP_START} ${TRAIN_END_DROP})

CHECKPOINT_DIR="results/rl_models/${MOBILITY}/drop_${TRAIN_END_DROP}_rx_UE_${RX_UES}_tx_UE_${NUM_TXUE_SEL}_imitation_none_steps_0"

echo "Training DEQN model for ${TRAIN_DROP_COUNT} drops (${TRAIN_DROP_START}-${TRAIN_END_DROP}) in a single run"
python sims/sim_mu_mimo_deqn_chan_pred_training.py \
    "${MOBILITY}" \
    "${TRAIN_DROPS}" \
    "${RX_UES}" \
    "${MODULATION_ORDER}" \
    "${CODE_RATE}" \
    "${NUM_TXUE_SEL}" \
    "${PERFECT_CSI}" \
    "deqn" \
    "${CSI_QUANTIZATION}" \
    "${LINK_ADAPT}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Expected checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

TEST_END_DROP=$((TEST_DROP_START + TEST_DROP_COUNT - 1))
echo "Testing with frozen model for drops ${TEST_DROP_START}-${TEST_END_DROP}"
for drop in $(seq ${TEST_DROP_START} ${TEST_END_DROP}); do
    echo "Running test drop ${drop}"
    python sims/sim_mu_mimo_testing_updates.py \
        "${MOBILITY}" \
        "${drop}" \
        "${RX_UES}" \
        "${MODULATION_ORDER}" \
        "${CODE_RATE}" \
        "${NUM_TXUE_SEL}" \
        "${PERFECT_CSI}" \
        "deqn" \
        "${CSI_QUANTIZATION}" \
        "${LINK_ADAPT}" \
        "${CHECKPOINT_DIR}" \
        "True"
done

echo "Training and testing complete."
