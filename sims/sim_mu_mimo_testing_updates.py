"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt

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

print(f"Script Name: {script_name}")
print(f"Arguments: {arguments}")

modulation_order = 2
code_rate = 2 / 3
num_txue_sel = 10
perfect_csi = True
channel_prediction_setting = "None"
csi_prediction = False
channel_prediction_method = None
csi_quantization_on = True

def _parse_bool(value):
    return str(value).lower() in ("true", "1", "yes")

def _parse_code_rate(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)

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

    if str(channel_prediction_setting).lower() == "none":
        csi_prediction = False
        channel_prediction_method = None
    else:
        csi_prediction = True
        channel_prediction_method = channel_prediction_setting

    if perfect_csi:
        csi_prediction = False
        channel_prediction_method = None
    
    print("Current mobility: {} \n Current drop: {} \n".format(mobility, drop_idx))
    print("rx_ues_arr: ", rx_ues_arr)
    print("rx_ues_arr[0]: ", rx_ues_arr[0])
    print("Modulation order: {}".format(modulation_order))
    print("Code rate: {}".format(code_rate))
    print("num_txue_sel: {}".format(num_txue_sel))
    print("perfect_csi: {}".format(perfect_csi))
    print("channel_prediction_setting: {}".format(channel_prediction_setting))
    print("csi_prediction: {}".format(csi_prediction))
    print("csi_quantization_on: {}".format(csi_quantization_on))
    print("channel_prediction_method: {}".format(channel_prediction_method))

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.rb_size = 12            # resource block size (this parameter is  currently only being used for ZF_QUANTIZED_CSI)
    cfg.total_slots = 100       # total number of slots in ns-3 channels
    cfg.start_slot_idx = 35     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.perfect_csi = perfect_csi
    cfg.rank_adapt = False      # enable/disable rank adaptation
    cfg.link_adapt = False      # enable/disable link adaptation,. .
    cfg.csi_prediction = csi_prediction
    cfg.use_perfect_csi_history_for_prediction = False
    cfg.channel_prediction_method = channel_prediction_method # "old", "two_mode", "two_mode_tf", "weiner_filter"
    cfg.enable_ue_selection = False
    cfg.scheduling = False
    if arguments == []:
        mobility = 'high_mobility'
        drop_idx = '3'
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
    # cfg.ns3_folder = "ns3/channels/LowMobility/"
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    cfg.estimated_channels_dir = "ns3/channel_estimates_" + mobility + "_drop_" + drop_idx
    cfg.enable_rxsquad = False
    cfg.precoding_method = "ZF" # Options: "ZF", "DIRECT" for quantized CSI feedback
    cfg.csi_quantization_on = csi_quantization_on
    cfg.PMI_feedback_architecture = 'dMIMO_phase2_type_II_CB2' # 'dMIMO_phase2_rel_15_type_II', 'dMIMO_phase2_type_II_CB1', 'dMIMO_phase2_type_II_CB2', 'RVQ'

    if cfg.perfect_csi:
        cfg.csi_prediction = False

    if cfg.link_adapt:
        MCS_string = "link_adapt"
    else:
        MCS_string = "mod_order_{}_code_rate_{}".format(modulation_order, code_rate)

    # Select Number of TxSquad and RxSquad UEs to use.
    ns3cfg.num_txue_sel = num_txue_sel
    if arguments == []:
        rx_ues_arr = [10]

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))    

    rc_config = RCConfig()
    rc_config.enable_window = True
    rc_config.window_length = 3
    rc_config.num_neurons = 16
    rc_config.history_len = 8    

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
    phase_1_ue_ber = []

    for ue_arr_idx in range(np.size(rx_ues_arr)):
        
        ns3cfg.num_rxue_sel = rx_ues_arr[ue_arr_idx]

        assert cfg.rank_adapt == False, "Current MU-MIMO code assumes fixed rank transmission (single stream per RX UE)."

        num_rx_antennas = rx_ues_arr[ue_arr_idx] * 2 + 4

        # Test case 1:  rank 2 transmission, assuming 2 antennas per UE and treating BS as two UEs
        # cfg.num_tx_streams = num_rx_antennas
        # cfg.ue_ranks = [2] # same rank for all UEs

        # Test case 2: rank 1 transmission, assuming 2 antennas per UE and treating BS as two UEs
        cfg.num_tx_streams = num_rx_antennas // 2
        cfg.ue_ranks = [1]  # same rank for all UEs

        # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
        cfg.modulation_order = modulation_order
        
        # if not cfg.scheduling:
        #     tx_ue_mask = np.zeros(10)
        #     tx_ue_mask[:ns3cfg.num_txue_sel] = np.ones(ns3cfg.num_txue_sel)
        #     rx_ue_mask = np.zeros(10)
        #     rx_ue_mask[:ns3cfg.num_rxue_sel] = np.ones(ns3cfg.num_rxue_sel)
        #     ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

        cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

        rst_zf = sim_mu_mimo_all(cfg, ns3cfg, rc_config)
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
            sinr_dB.append(np.concatenate(rst_zf[11]))

        folder_path = "results/channels_multiple_mu_mimo/{}".format(folder_name)
        os.makedirs(folder_path, exist_ok=True)

        if cfg.csi_prediction:
            
            if cfg.scheduling:
                file_path = os.path.join(folder_path, "mu_mimo_results_{}_scheduling_tx_UE_{}_prediction_{}_pmi_quantization_{}.npz".format(MCS_string, num_txue_sel, cfg.channel_prediction_method, cfg.csi_quantization_on))
            else:
                file_path = os.path.join(folder_path, "mu_mimo_results_{}_rx_UE_{}_tx_UE_{}_prediction_{}_pmi_quantization_{}.npz".format(MCS_string, rx_ues_arr[ue_arr_idx], num_txue_sel, cfg.channel_prediction_method, cfg.csi_quantization_on))
            np.savez(file_path,
                    cfg=cfg, ns3cfg=ns3cfg, ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_zf[5],
                    nodewise_throughput=rst_zf[6], nodewise_bitrate=rst_zf[7], ranks=rst_zf[8], uncoded_ber_list=rst_zf[9],
                    ldpc_ber_list=rst_zf[10], sinr_dB=rst_zf[11])
        else:
            if cfg.scheduling:
                file_path = os.path.join(folder_path, "mu_mimo_results_{}_scheduling_tx_UE_{}_perfect_CSI_{}_pmi_quantization_{}.npz".format(MCS_string, num_txue_sel, cfg.perfect_csi, cfg.csi_quantization_on))
            else:
                file_path = os.path.join(folder_path, "mu_mimo_results_{}_rx_UE_{}_tx_UE_{}_perfect_CSI_{}_pmi_quantization_{}.npz".format(MCS_string, rx_ues_arr[ue_arr_idx], num_txue_sel, cfg.perfect_csi, cfg.csi_quantization_on))

            np.savez(file_path,
                    cfg=cfg, ns3cfg=ns3cfg, ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, 
                    nodewise_goodput=rst_zf[5], nodewise_throughput=rst_zf[6], nodewise_bitrate=rst_zf[7], 
                    ranks=rst_zf[8], uncoded_ber_list=rst_zf[9], ldpc_ber_list=rst_zf[10], sinr_dB=rst_zf[11])
