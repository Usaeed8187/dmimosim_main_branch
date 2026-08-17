#!/bin/bash

# Array of arguments
declare -a mobilities=("highest_mobility")
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

# One CSI report becomes available after one complete D-MIMO cycle. Select
# either profile (or both). The simulation validates that
# feedback_delay = num_slots_p1 + num_slots_p2.
declare -a feedback_delays_ms=("4" "8")

# Residual synchronization sweep under the clock_v2 model. Each entry is
# "fractional-frequency std (ppb), initial-timing std (ps), initial-phase std
# (degrees), phase-noise S100 (dBc/Hz or off)". Merlo et al. report 3.73 ppb
# frequency RMSE and roughly 60--70 ps coordination precision; the surrounding
# points bracket those demonstrated values. Ngo--Larsson use S100 in
# [-120,-80] dBc/Hz. The explicit all-zero entry ensures that the baseline is
# generated with the same mobility, drop set, and code revision as the sweep.
declare -a sync_error_pairs=(
    "0 0 0 off"
    # "1 0 0 off"
    # "3.73 0 0 off"
    # "10 0 0 off"
    # "30 0 0 off"
    # "0 30 0 off"
    # "0 60 0 off"
    # "0 70 0 off"
    # "0 200 0 off"
    "0 0 0 -120"
    "0 0 0 -110"
    "0 0 0 -100"
    "0 0 0 -90"
    "0 0 0 -80"
)

# # Preserve the former experiment as an explicitly labeled stress profile,
# # converted to the new physical units at fc=3.5 GHz, fs=7.68 MHz, and 1 ms.
# SYNC_SWEEP_PROFILE=${SYNC_SWEEP_PROFILE:-paper}
# if [[ "${SYNC_SWEEP_PROFILE}" == "stress" ]]; then
#     sync_error_pairs=(
#         "2.857142857 0 0 off"
#         "14.285714286 0 0 off"
#         "28.571428571 0 0 off"
#         "35.714285714 0 0 off"
#         "71.428571429 0 0 off"
#         "0 6510.416667 0 off"
#         "0 13020.833333 0 off"
#         "0 26041.666667 0 off"
#         "0 65104.166667 0 off"
#     )
# elif [[ "${SYNC_SWEEP_PROFILE}" != "paper" ]]; then
#     echo "Unsupported SYNC_SWEEP_PROFILE=${SYNC_SWEEP_PROFILE}; use paper or stress." >&2
#     exit 2
# fi

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
export BALANCED_LITE_HANKEL_ENERGY=${BALANCED_LITE_HANKEL_ENERGY:-0.80}
export PREDICTOR_WORKERS=${PREDICTOR_WORKERS:-8}
# Link-level parallelism supplies the concurrency. Prevent every worker from
# independently starting a full OpenBLAS thread team.
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export DMIMO_PHASE_1_ENABLED=False
export DMIMO_PHASE_3_ENABLED=False


feedback_profile() {
    case "$1" in
        4)
            echo "2 2 8 0.5"
            ;;
        8)
            echo "4 4 8 0.6666666666666666"
            ;;
        *)
            echo "Unsupported feedback delay '$1' ms; use 4 and/or 8." >&2
            return 2
            ;;
    esac
}

for feedback_delay_ms in "${feedback_delays_ms[@]}"; do
    feedback_profile "${feedback_delay_ms}" >/dev/null || exit $?
done

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

                                        for feedback_delay_ms in "${feedback_delays_ms[@]}"; do
                                            profile=$(feedback_profile "${feedback_delay_ms}") || exit $?
                                            read -r num_slots_p1 num_slots_p2 start_slot_idx offline_ratio <<< "${profile}"
                                            for sync_error_pair in "${sync_error_pairs[@]}"; do
                                                read -r freq_std_ppb timing0_std_ps phase0_std_deg pn_s100_dbchz <<< "${sync_error_pair}"
                                                sync_errors="True"
                                                echo "Mobility: ${mobilities[$i]}, Drop idx: ${drop_idx[$j]}, Rx UEs: ${rx_ues_arr[$k]}, channel_prediction_method: ${channel_prediction_method}, feedback_delay_ms: ${feedback_delay_ms}, P1/P2: ${num_slots_p1}/${num_slots_p2}, start_slot_idx: ${start_slot_idx}, offline_ratio: ${offline_ratio}, freq_std_ppb: ${freq_std_ppb}, timing0_std_ps: ${timing0_std_ps}, phase0_std_deg: ${phase0_std_deg}, pn_s100_dbchz: ${pn_s100_dbchz}" >&2
                                                echo "${mobilities[$i]} ${drop_idx[$j]} ${rx_ues_arr[$k]} ${modulation_orders[$m]} ${code_rates[$c]} ${num_txue_sel_arr[$t]} ${perfect_csi_arr[$pcsi]} ${channel_prediction_setting} ${csi_quantization_arr[$cquant]} ${link_adapt} ${sync_errors} ${freq_std_ppb} ${timing0_std_ps} ${phase0_std_deg} ${pn_s100_dbchz} ${feedback_delay_ms} ${num_slots_p1} ${num_slots_p2} ${start_slot_idx} ${offline_ratio}"
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
    done
}

mapfile -t scenario_args < <(generate_args)

total_scenarios=${#scenario_args[@]}
if [[ "${DRY_RUN:-False}" == "True" ]]; then
    printf '%s\n' "${scenario_args[@]}"
    echo "Dry run: ${total_scenarios} scenarios" >&2
    exit 0
fi
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
    local feedback_delay_ms=${args[15]}
    local num_slots_p1=${args[16]}
    local num_slots_p2=${args[17]}
    local start_slot_idx=${args[18]}
    local offline_ratio=${args[19]}
    env \
        DMIMO_GEN_SYNC_ERRORS="${sync_errors}" \
        DMIMO_SYNC_FREQ_STD_PPB="${freq_std_ppb}" \
        DMIMO_SYNC_INITIAL_TIMING_STD_PS="${timing0_std_ps}" \
        DMIMO_SYNC_INITIAL_PHASE_STD_DEG="${phase0_std_deg}" \
        DMIMO_SYNC_PHASE_NOISE_S100_DBCHZ="${pn_s100_dbchz}" \
        DMIMO_CSI_FEEDBACK_DELAY_MS="${feedback_delay_ms}" \
        DMIMO_NUM_SLOTS_P1="${num_slots_p1}" \
        DMIMO_NUM_SLOTS_P2="${num_slots_p2}" \
        DMIMO_START_SLOT_IDX="${start_slot_idx}" \
        DMIMO_WESN_OFFLINE_RATIO="${offline_ratio}" \
        python sims/sim_mu_mimo_testing_updates.py "${args[@]:0:10}"
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
