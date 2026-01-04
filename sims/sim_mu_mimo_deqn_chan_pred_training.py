"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import datetime
import traceback
import math
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from typing import List, Optional
import time

gpu_num = 0  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm/16/lib64/libLLVM.so'

import tensorflow as tf

dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)
# Configure to use only a single GPU and allocate only as much memory as needed
gpus = tf.config.list_physical_devices('GPU')
if gpus and gpu_num != "":
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

from dmimo.config import SimConfig, Ns3Config, RCConfig
from dmimo.mu_mimo_testing_updates import sim_mu_mimo_all
from dmimo.channel.rl_beam_selector import RLBeamSelector
from sionna.ofdm import ResourceGrid
from dmimo.channel import LMMSELinearInterp, dMIMOChannels, estimate_freq_cov

# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

# Set symlinks to channel data files
sys.path.append(os.path.join('..'))
source_dir = '/home/usama/ns3_channels_q4/'
destination_dir = 'ns3/'
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
for root, dirs, files in os.walk(source_dir):
    # Construct the relative path to replicate the directory structure
    relative_path = os.path.relpath(root, source_dir)
    destination_subdir = os.path.join(destination_dir, relative_path)

    # Create the subdirectory in the destination if it doesn't exist
    if not os.path.exists(destination_subdir):
        os.makedirs(destination_subdir)
    
    # Create symlinks for each file in the current directory
    for file in files:
        source_file = os.path.join(root, file)
        destination_file = os.path.join(destination_subdir, file)

        # If the symlink already exists, remove it
        if os.path.exists(destination_file):
            os.remove(destination_file)

        # Create the symlink
        os.symlink(source_file, destination_file)
        # print(f"Symlink created for {source_file} -> {destination_file}")

script_name = sys.argv[0]
arguments = sys.argv[1:]

mobility = 'high_mobility'
# drop_idx = '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24'
drop_idx = ','.join(str(i) for i in range(1, 101))
drop_list: List[str] = [item.strip() for item in drop_idx.split(',') if item.strip()]
rx_ues_arr = [2]
num_txue_sel = 2

modulation_order = 4
code_rate = 1 / 2
link_adapt = True

perfect_csi = False
channel_prediction_setting = "deqn" # "None", "two_mode", "weiner_filter", "deqn"
csi_prediction = True
channel_prediction_method = "deqn" # None, "two_mode", "weiner_filter", "deqn"
csi_quantization_on = True
imitation_method = "two_mode" # "none", "weiner_filter", "two_mode"
imitation_drop_count = 10

def _build_imitation_info() -> Optional[str]:
    if imitation_method == "none" or imitation_drop_count <= 0:
        return None

    return (
        "imitation learning enabled "
        f"(method={imitation_method}, drop_count={imitation_drop_count})"
    )

def _build_imitation_tag() -> str:
    """Return a filesystem-safe tag describing the imitation configuration."""

    method = (imitation_method or "none").replace(" ", "_").lower()
    steps = max(imitation_drop_count, 0)
    return f"imitation_{method}_steps_{steps}"

def log_error(exc: Exception) -> str:
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("results", "logs", f"sim_error_{timestamp}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Simulation halted due to an error:\n")
        traceback.print_exception(exc, file=log_file)

    print(f"Error encountered. Details written to {log_path}")
    return log_path

def _parse_bool(value):
    return str(value).lower() in ("true", "1", "yes")

def _parse_code_rate(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)
    
def _parse_drop_indices(raw_drop_value: str) -> List[str]:
    """Parse drop strings that may include comma-separated lists or ranges."""

    values: List[str] = []
    for part in str(raw_drop_value).split(','):
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            start_str, end_str = part.split('-', maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            values.extend(str(i) for i in range(start, end + step, step))
        else:
            values.append(part)

    return values

def parse_arguments():
    global mobility, drop_idx, rx_ues_arr, drop_list
    global modulation_order, code_rate, num_txue_sel
    global perfect_csi, channel_prediction_setting
    global csi_prediction, channel_prediction_method
    global csi_quantization_on, link_adapt
    global imitation_method, imitation_drop_count

    if len(arguments) > 0:
        mobility = arguments[0]
        drop_idx = arguments[1]
        rx_ues_arr = [arguments[2]]
        rx_ues_arr = np.array(rx_ues_arr, dtype=int)

        if len(arguments) >= 4:
            modulation_order = int(arguments[3])

        if len(arguments) >= 5:
            code_rate = _parse_code_rate(arguments[4])
        
        if len(arguments) >= 6:
            num_txue_sel = int(arguments[5])
        
        if len(arguments) >= 7:
            perfect_csi = _parse_bool(arguments[6])
        
        if len(arguments) >= 8:
            channel_prediction_setting = arguments[7]

        if len(arguments) >= 9:
            csi_quantization_on = _parse_bool(arguments[8])

        if len(arguments) >= 10:
            link_adapt = _parse_bool(arguments[9])

        if len(arguments) >= 11:
            imitation_method = str(arguments[10]).lower()

        if len(arguments) >= 12:
            imitation_drop_count = int(arguments[11])

        if str(channel_prediction_setting).lower() == "none":
            csi_prediction = False
            channel_prediction_method = None
        else:
            csi_prediction = True
            channel_prediction_method = channel_prediction_setting

        if perfect_csi:
            csi_prediction = False
            channel_prediction_method = None

        drop_list = _parse_drop_indices(drop_idx)

        print("Current mobility: {} \n Current drop: {} \n".format(mobility, drop_idx))
        # print("rx_ues_arr: ", rx_ues_arr)
        # print("rx_ues_arr[0]: ", rx_ues_arr[0])
        # print("Modulation order: {}".format(modulation_order))
        # print("Code rate: {}".format(code_rate))
        # print("num_txue_sel: {}".format(num_txue_sel))
        # print("perfect_csi: {}".format(perfect_csi))
        # print("channel_prediction_setting: {}".format(channel_prediction_setting))
        # print("csi_prediction: {}".format(csi_prediction))
        # print("csi_quantization_on: {}".format(csi_quantization_on))
        # print("channel_prediction_method: {}".format(channel_prediction_method))
        # print("link_adapt: {}".format(link_adapt))
    else:
        drop_list = _parse_drop_indices(drop_idx)

def _build_model_dir(drop_label: str, rx_ue_sel: int, imitation_tag: str) -> str:
    return os.path.join(
        "results",
        "rl_models",
        mobility,
        f"drop_{drop_label}_rx_UE_{rx_ue_sel}_tx_UE_{num_txue_sel}_{imitation_tag}",
    )

def _try_resume_from_checkpoint(
    rl_selector: Optional[RLBeamSelector], rx_ue_sel: int, imitation_tag: str
) -> None:
    if rl_selector is None:
        return

    try:
        first_drop = int(drop_list[0]) if drop_list else None
    except ValueError:
        first_drop = None

    if first_drop is None or first_drop <= 1:
        return

    for candidate in range(first_drop - 1, 0, -1):
        model_dir = _build_model_dir(str(candidate), rx_ue_sel, imitation_tag)
        if os.path.isdir(model_dir):
            print(f"Loading DEQN checkpoint from {model_dir}")
            rl_selector.load_all(model_dir)
            return

    print("No earlier DEQN checkpoint found. Starting fresh training session.")

# Main function
def run_simulation():
    global mobility, drop_idx, rx_ues_arr, drop_list
    parse_arguments()

    imitation_info = _build_imitation_info()
    imitation_tag = _build_imitation_tag()

    rc_config = RCConfig()
    rc_config.enable_window = True
    rc_config.window_length = 3
    rc_config.num_neurons = 16
    rc_config.history_len = 8

    shared_rl_selector = (
        RLBeamSelector(imitation_method=imitation_method) if channel_prediction_method == "deqn" else None
    )

    shared_rl_selector_2 = (
        RLBeamSelector(imitation_method=imitation_method) if channel_prediction_method == "deqn" else None
    )

    _try_resume_from_checkpoint(shared_rl_selector, int(rx_ues_arr[0]), imitation_tag)
    _try_resume_from_checkpoint(shared_rl_selector_2, int(rx_ues_arr[0]), imitation_tag)

    for drop_number, drop_idx in enumerate(drop_list, start=1):
        start_time = time.time()
        # Simulation settings
        cfg = SimConfig()
        cfg.rb_size = 12            # resource block size (this parameter is  currently only being used for ZF_QUANTIZED_CSI)
        cfg.total_slots = 99       # total number of slots in ns-3 channels
        cfg.start_slot_idx = 35     # starting slots (must be greater than csi_delay + 5)
        cfg.csi_delay = 4           # feedback delay in number of subframe
        cfg.perfect_csi = perfect_csi
        cfg.rank_adapt = False      # enable/disable rank adaptation
        cfg.link_adapt = link_adapt      # enable/disable link adaptation,. .
        cfg.csi_prediction = csi_prediction
        cfg.use_perfect_csi_history_for_prediction = False
        cfg.channel_prediction_method = channel_prediction_method # "old", "two_mode", "two_mode_tf", "weiner_filter"
        cfg.enable_ue_selection = False
        cfg.scheduling = False
        cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
        ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
        cfg.estimated_channels_dir = "ns3/channel_estimates_" + mobility + "_drop_" + drop_idx
        cfg.enable_rxsquad = False
        cfg.precoding_method = "ZF" # Options: "ZF", "DIRECT", "SLNR" for quantized CSI feedback
        cfg.csi_quantization_on = csi_quantization_on
        cfg.PMI_feedback_architecture = 'dMIMO_phase2_type_II_CB2' # 'dMIMO_phase2_rel_15_type_II', 'dMIMO_phase2_type_II_CB1', 'dMIMO_phase2_type_II_CB2', 'RVQ'
        cfg.lmmse_cov_est_slots = 5  # Number of slots to use for channel covariance estimation
        warm_start_active = (
            channel_prediction_method == "deqn"
            and imitation_method != "none"
            and drop_number <= imitation_drop_count
        )
        cfg.imitation_method = imitation_method if warm_start_active else "none"
        cfg.use_imitation_override = warm_start_active
        cfg.imitation_drop_count = imitation_drop_count

        if shared_rl_selector is not None:
            time_steps_per_drop = math.ceil(
                (cfg.total_slots - cfg.start_slot_idx)
                / (cfg.num_slots_p1 + cfg.num_slots_p2)
            ) - 1
            epsilon_total_steps = len(drop_list) * time_steps_per_drop
            shared_rl_selector.set_epsilon_total_steps(epsilon_total_steps)
            shared_rl_selector_2.set_epsilon_total_steps(epsilon_total_steps)

        if cfg.perfect_csi:
            cfg.csi_prediction = False

        if cfg.link_adapt:
            MCS_string = "link_adapt"
        else:
            MCS_string = "mod_order_{}_code_rate_{}".format(modulation_order, code_rate)

        ns3cfg.num_txue_sel = num_txue_sel

        folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
        os.makedirs(os.path.join("results", folder_name), exist_ok=True)

        folder_path = "results/channels_multiple_mu_mimo/{}".format(folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Precompute LMMSE resources once per drop when needed
        if not cfg.perfect_csi:
            num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant

            csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
            csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
            csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

            rg_csi = ResourceGrid(
                num_ofdm_symbols=14,
                fft_size=cfg.fft_size,
                subcarrier_spacing=cfg.subcarrier_spacing,
                num_tx=1,
                num_streams_per_tx=num_txs_ant,
                cyclic_prefix_length=cfg.cyclic_prefix_len,
                num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                dc_null=False,
                pilot_pattern="kronecker",
                pilot_ofdm_symbol_indices=[2, 11],
            )

            dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
            slot_idx = cfg.start_slot_idx - cfg.csi_delay
            cache_slots = (cfg.lmmse_cov_est_slots if slot_idx >= cfg.lmmse_cov_est_slots else slot_idx)
            start_slot = slot_idx - cache_slots + 1

            freq_cov_mat = estimate_freq_cov(dmimo_chans, rg_csi, start_slot=start_slot, total_slots=cache_slots)
            lmmse_interpolator = LMMSELinearInterp(rg_csi.pilot_pattern, freq_cov_mat)

            cfg.freq_cov_mat = freq_cov_mat
            cfg.lmmse_interpolator = lmmse_interpolator

        #############################################
        # Testing
        #############################################

        ber = np.zeros(np.size(rx_ues_arr ))
        ldpc_ber = np.zeros(np.size(rx_ues_arr ))
        goodput = np.zeros(np.size(rx_ues_arr ))
        throughput = np.zeros(np.size(rx_ues_arr ))
        bitrate = np.zeros(np.size(rx_ues_arr ))
        nodewise_goodput = []
        nodewise_throughput = []
        nodewise_bitrate = []
        ranks = []
        ldpc_ber_list = []
        uncoded_ber_list = []
        sinr_dB = []
        snr_dB = []
        phase_1_ue_ber = []

        last_rx_ue_sel = None

        for ue_arr_idx in range(np.size(rx_ues_arr)):

            ns3cfg.num_rxue_sel = rx_ues_arr[ue_arr_idx]
            last_rx_ue_sel = ns3cfg.num_rxue_sel

            assert cfg.rank_adapt == False, "Current MU-MIMO code assumes fixed rank transmission (single stream per RX UE)."

            num_rx_antennas = rx_ues_arr[ue_arr_idx] * 2 + 4

            cfg.num_tx_streams = num_rx_antennas // 2
            cfg.ue_ranks = [1]  # same rank for all UEs

            cfg.modulation_order = modulation_order
            cfg.code_rate = code_rate

            cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

            rst_zf = sim_mu_mimo_all(cfg, ns3cfg, rc_config, rl_selector=shared_rl_selector, rl_selector_2=shared_rl_selector_2)
            ber[ue_arr_idx] = rst_zf[0]
            ldpc_ber[ue_arr_idx] = rst_zf[1]
            goodput[ue_arr_idx] = rst_zf[2]
            throughput[ue_arr_idx] = rst_zf[3]
            bitrate[ue_arr_idx] = rst_zf[4]

            nodewise_goodput.append(rst_zf[5])
            nodewise_throughput.append(rst_zf[6])
            nodewise_bitrate.append(rst_zf[7])
            ranks.append(rst_zf[8])
            uncoded_ber_list.append(rst_zf[9])
            ldpc_ber_list.append(rst_zf[10])
            if rst_zf[11] is not None:
                sinr_dB.append(rst_zf[11])
            if rst_zf[12] is not None:
                snr_dB.append(rst_zf[12])

            if cfg.csi_prediction:

                if cfg.scheduling:
                    file_path = os.path.join(folder_path, "mu_mimo_results_{}_scheduling_tx_UE_{}_prediction_{}_pmi_quantization_{}_{}.npz".format(MCS_string, num_txue_sel, cfg.channel_prediction_method, cfg.csi_quantization_on, imitation_tag))
                else:
                    file_path = os.path.join(folder_path, "mu_mimo_results_{}_rx_UE_{}_tx_UE_{}_prediction_{}_pmi_quantization_{}_{}.npz".format(MCS_string, rx_ues_arr[ue_arr_idx], num_txue_sel, cfg.channel_prediction_method, cfg.csi_quantization_on, imitation_tag))
                npz_payload = {
                    "cfg": cfg,
                    "ns3cfg": ns3cfg,
                    "ber": ber,
                    "ldpc_ber": ldpc_ber,
                    "goodput": goodput,
                    "throughput": throughput,
                    "bitrate": bitrate,
                    "nodewise_goodput": rst_zf[5],
                    "nodewise_throughput": rst_zf[6],
                    "nodewise_bitrate": rst_zf[7],
                    "ranks": rst_zf[8],
                    "uncoded_ber_list": rst_zf[9],
                    "ldpc_ber_list": rst_zf[10],
                    "sinr_dB": rst_zf[11],
                    "snr_dB": rst_zf[12],
                }

                if imitation_info:
                    npz_payload["imitation_info"] = imitation_info

                np.savez(file_path, **npz_payload)
            else:
                if cfg.scheduling:
                    file_path = os.path.join(folder_path, "mu_mimo_results_{}_scheduling_tx_UE_{}_perfect_CSI_{}_pmi_quantization_{}_{}.npz".format(MCS_string, num_txue_sel, cfg.perfect_csi, cfg.csi_quantization_on, imitation_tag))
                else:
                    file_path = os.path.join(folder_path, "mu_mimo_results_{}_rx_UE_{}_tx_UE_{}_perfect_CSI_{}_pmi_quantization_{}_{}.npz".format(MCS_string, rx_ues_arr[ue_arr_idx], num_txue_sel, cfg.perfect_csi, cfg.csi_quantization_on, imitation_tag))
                
                npz_payload = {
                    "cfg": cfg,
                    "ns3cfg": ns3cfg,
                    "ber": ber,
                    "ldpc_ber": ldpc_ber,
                    "goodput": goodput,
                    "throughput": throughput,
                    "bitrate": bitrate,
                    "nodewise_goodput": rst_zf[5],
                    "nodewise_throughput": rst_zf[6],
                    "nodewise_bitrate": rst_zf[7],
                    "ranks": rst_zf[8],
                    "uncoded_ber_list": rst_zf[9],
                    "ldpc_ber_list": rst_zf[10],
                    "sinr_dB": rst_zf[11],
                    "snr_dB": rst_zf[12],
                }

                if imitation_info:
                    npz_payload["imitation_info"] = imitation_info

                np.savez(file_path, **npz_payload)
        
        if shared_rl_selector is not None:
            if last_rx_ue_sel is None:
                raise RuntimeError("RX UE selection was not set before saving rewards.")

            rewards = np.array(shared_rl_selector.get_reward_log(), dtype=np.float32)
            rewards_path = os.path.join(
                folder_path,
                f"deqn_rewards_drop_{drop_idx}_rx_UE_{last_rx_ue_sel}_tx_UE_{num_txue_sel}_{imitation_tag}.npz",
            )
            rewards_payload = {"rewards": rewards}
            if imitation_info:
                rewards_payload["imitation_info"] = imitation_info

            np.savez(rewards_path, **rewards_payload)
            print(f"Saved DEQN rewards to {rewards_path}")

            actions = np.array(shared_rl_selector.get_action_log(), dtype=np.int64)
            actions_path = os.path.join(
                folder_path,
                f"deqn_actions_drop_{drop_idx}_rx_UE_{last_rx_ue_sel}_tx_UE_{num_txue_sel}_{imitation_tag}.npz",
            )
            actions_payload = {"actions": actions}
            if imitation_info:
                actions_payload["imitation_info"] = imitation_info

            np.savez(actions_path, **actions_payload)
            print(f"Saved DEQN actions to {actions_path}")

        if shared_rl_selector is not None:
            if last_rx_ue_sel is None:
                raise RuntimeError("RX UE selection was not set before saving models.")
            model_dir = _build_model_dir(drop_idx, last_rx_ue_sel, imitation_tag)
            shared_rl_selector.save_all(model_dir, imitation_info=imitation_info)
            print(f"Saved DEQN models to {model_dir}")

            shared_rl_selector.reset_episode()
        end_time = time.time()
        print("Drop {} simulation time: {} seconds".format(drop_idx, end_time - start_time))


if __name__ == "__main__":
    # try:
    #     run_simulation()
    # except Exception as exc:  # noqa: BLE001
    #     log_error(exc)
    #     sys.exit(1)

    run_simulation()