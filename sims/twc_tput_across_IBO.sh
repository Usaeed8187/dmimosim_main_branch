#!/bin/bash

declare -a mobilities=("higher_mobility")
declare -a drop_idx=($(seq 1 20))
declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("8")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a channel_prediction_settings=(
    "configured_wesn_balanced"
    "configured_wesn_balanced_lite"
    "kalman_filter"
    "steady_state_kalman_filter"
)
declare -a csi_quantization_arr=("True")
# declare -a ibo_db_values=("0" "3" "5" "6.5" "9")
declare -a ibo_db_values=("0" "3")

link_adapt="True"
PARALLEL_JOBS=${PARALLEL_JOBS:-6}

export WESN_LITE_ESN_K=${WESN_LITE_ESN_K:-4}
export WESN_LITE_RESIDUE_ENERGY=${WESN_LITE_RESIDUE_ENERGY:-0.9}
export WESN_LITE_READOUT_MODE=${WESN_LITE_READOUT_MODE:-centered_ridge}
export WESN_LITE_SUBCARRIERS_PER_RB=${WESN_LITE_SUBCARRIERS_PER_RB:-12}
export BALANCED_LITE_HANKEL_ENERGY=${BALANCED_LITE_HANKEL_ENERGY:-0.80}
export PREDICTOR_WORKERS=${PREDICTOR_WORKERS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export DMIMO_PHASE_1_ENABLED=False
export DMIMO_PHASE_3_ENABLED=False
export DMIMO_NUM_SLOTS_P1=2
export DMIMO_NUM_SLOTS_P2=2
export DMIMO_GEN_SYNC_ERRORS=False
export DMIMO_SYNC_FREQ_STD_PPB=0
export DMIMO_SYNC_INITIAL_TIMING_STD_PS=0
export DMIMO_SYNC_INITIAL_PHASE_STD_DEG=0
export DMIMO_SYNC_PHASE_NOISE_S100_DBCHZ=off
export DMIMO_PA_ENABLED=True
export DMIMO_PA_RHO=${DMIMO_PA_RHO:-3}
export DMIMO_PA_MODEL_VERSION=${DMIMO_PA_MODEL_VERSION:-rapp_v1}

generate_args() {
    for mobility in "${mobilities[@]}"; do
        for drop in "${drop_idx[@]}"; do
            for rx_ues in "${rx_ues_arr[@]}"; do
                for mod_order in "${modulation_orders[@]}"; do
                    for code_rate in "${code_rates[@]}"; do
                        for tx_ues in "${num_txue_sel_arr[@]}"; do
                            for perfect_csi in "${perfect_csi_arr[@]}"; do
                                for method in "${channel_prediction_settings[@]}"; do
                                    for quantization in "${csi_quantization_arr[@]}"; do
                                        for ibo_db in "${ibo_db_values[@]}"; do
                                            echo "Mobility: ${mobility}, Drop: ${drop}, Rx UEs: ${rx_ues}, Tx UEs: ${tx_ues}, method: ${method}, IBO: ${ibo_db} dB" >&2
                                            echo "${mobility} ${drop} ${rx_ues} ${mod_order} ${code_rate} ${tx_ues} ${perfect_csi} ${method} ${quantization} ${link_adapt} ${ibo_db}"
                                        done
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
    local ibo_db=${args[10]}
    env DMIMO_PA_IBO_DB="${ibo_db}" \
        python sims/sim_mu_mimo_testing_updates.py "${args[@]:0:10}"
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

echo "All ${completed_jobs} scenarios completed" >&2

if [[ "${PLOT_AFTER_RUN:-True}" == "True" ]]; then
    python results/plot_results_twc_chanpred.py \
        --plot-pa-sweep \
        --mobility "${mobilities[0]}" \
        --drops "${drop_idx[@]}" \
        --rx-ues "${rx_ues_arr[@]}" \
        --tx-ues "${num_txue_sel_arr[@]}" \
        --fixed-rx "${rx_ues_arr[0]}" \
        --fixed-tx "${num_txue_sel_arr[0]}" \
        --modulation-orders "${modulation_orders[@]}" \
        --code-rates "${code_rates[@]}" \
        --pa-ibo-db-values "${ibo_db_values[@]}" \
        --pa-rho "${DMIMO_PA_RHO}" \
        --pa-model-version "${DMIMO_PA_MODEL_VERSION}"
fi
