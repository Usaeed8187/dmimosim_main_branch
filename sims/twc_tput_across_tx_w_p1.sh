#!/bin/bash

# Array of arguments
# declare -a mobilities=("low_mobility" "medium_mobility" "high_mobility")
declare -a mobilities=("higher_mobility" "highest_mobility" "high_mobility")
# declare -a drop_idx=("26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36" "37" "38" "39" "43" "44" "45")
# declare -a drop_idx=("1")
declare -a drop_idx=($(seq 1 20))
declare -a rx_ues_arr=("4")
declare -a num_txue_sel_arr=("2" "4" "6" "8" "10")
declare -a modulation_orders=("4")
declare -a code_rates=("1/2")
declare -a perfect_csi_arr=("False")
declare -a channel_prediction_settings=("configured_wesn_balanced" "configured_wesn_balanced_lite" "kalman_filter" "steady_state_kalman_filter") # Phase-2 predictors; configured identically to the phase-2-only sweep.
declare -a csi_quantization_arr=("True")

declare -a sync_error_settings=(
    "False 0 0 0 off"
    "True 1 0 0 off"
    "True 3.73 0 0 off"
    "True 10 0 0 off"
    "True 30 0 0 off"
    "True 0 30 0 off"
    "True 0 60 0 off"
    "True 0 70 0 off"
    "True 0 200 0 off"
    "True 0 0 0 -120"
    "True 0 0 0 -100"
    "True 0 0 0 -90"
    "True 0 0 0 -80"
)

link_adapt="True"

if [[ "${link_adapt}" == "True" ]]; then
    modulation_orders=("${modulation_orders[0]}")
    code_rates=("${code_rates[0]}")
fi

PARALLEL_JOBS=${PARALLEL_JOBS:-6}

# WESN-Lite settings for the throughput sweep. Callers may override any of
# these through the environment.
export WESN_LITE_ESN_K=${WESN_LITE_ESN_K:-4}
export WESN_LITE_RESIDUE_ENERGY=${WESN_LITE_RESIDUE_ENERGY:-0.9}
export WESN_LITE_READOUT_MODE=${WESN_LITE_READOUT_MODE:-centered_ridge}
export WESN_LITE_SUBCARRIERS_PER_RB=${WESN_LITE_SUBCARRIERS_PER_RB:-12}
export BALANCED_LITE_HANKEL_ENERGY=${BALANCED_LITE_HANKEL_ENERGY:-0.90}
export PREDICTOR_WORKERS=${PREDICTOR_WORKERS:-8}
# Link-level parallelism supplies the concurrency. Prevent every worker from
# independently starting a full OpenBLAS thread team.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export DMIMO_PHASE_1_ENABLED=True
export DMIMO_PHASE_3_ENABLED=False
export DMIMO_NUM_SLOTS_P1=2
export DMIMO_NUM_SLOTS_P2=2

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

                                        for sync_setting in "${sync_error_settings[@]}"; do
                                            read -r sync_errors freq_std_ppb timing0_std_ps phase0_std_deg pn_s100_dbchz <<< "${sync_setting}"
                                            echo "${mobilities[$i]} ${drop_idx[$j]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi_arr[$pcsi]} ${channel_prediction_setting} ${csi_quantization_arr[$cquant]} ${link_adapt} ${sync_errors} ${freq_std_ppb} ${timing0_std_ps} ${phase0_std_deg} ${pn_s100_dbchz}"
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
    local sync_errors=${args[10]}
    local freq_std_ppb=${args[11]}
    local timing0_std_ps=${args[12]}
    local phase0_std_deg=${args[13]}
    local pn_s100_dbchz=${args[14]}
    env \
        DMIMO_GEN_SYNC_ERRORS="${sync_errors}" \
        DMIMO_SYNC_FREQ_STD_PPB="${freq_std_ppb}" \
        DMIMO_SYNC_INITIAL_TIMING_STD_PS="${timing0_std_ps}" \
        DMIMO_SYNC_INITIAL_PHASE_STD_DEG="${phase0_std_deg}" \
        DMIMO_SYNC_PHASE_NOISE_S100_DBCHZ="${pn_s100_dbchz}" \
        python sims/sim_mu_mimo_testing_updates_w_p1.py "${args[@]:0:10}"
}

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
