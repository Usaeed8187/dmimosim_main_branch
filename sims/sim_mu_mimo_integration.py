"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

gpu_num = 0  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm/16/lib64/libLLVM.so'

# Configure to use only a single GPU and allocate only as much memory as needed
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus and gpu_num != "":
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

sys.path.append(os.path.join('..'))
source_dir = '/home/data/ns3_channels_q4/'
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

from dmimo.config import SimConfig, Ns3Config
from dmimo.mu_mimo import sim_mu_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.rank_adapt = True      # disable rank adaptation
    cfg.link_adapt = True      # disable link adaptation
    cfg.csi_prediction = False
    mobility = 'medium_mobility'
    drop_idx = '3'
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))
    
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    # rx_ues_arr = [1,2,4,6]
    rx_ues_arr = [1]    

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

    #############################################
    # Testing with rank and link adaptation
    #############################################

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
        ns3cfg.num_rxue_sel = cfg.num_rx_ue_sel
        cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

        cfg.precoding_method = "ZF"
        rst_bd = sim_mu_mimo_all(cfg, ns3cfg)
        ber[ue_arr_idx] = rst_bd[0]
        ldpc_ber[ue_arr_idx] = rst_bd[1]
        goodput[ue_arr_idx] = rst_bd[2]
        throughput[ue_arr_idx] = rst_bd[3]
        bitrate[ue_arr_idx] = rst_bd[4]
        phase_1_ue_ber_tmp = rst_bd[12]
        
        nodewise_goodput.append(rst_bd[5])
        nodewise_throughput.append(rst_bd[6])
        nodewise_bitrate.append(rst_bd[7])
        ranks.append(rst_bd[8])
        uncoded_ber_list.append(rst_bd[9])
        ldpc_ber_list.append(rst_bd[10])
        if rst_bd[11] is not None:
            sinr_dB.append(np.concatenate(rst_bd[11]))
        phase_1_ue_ber.append(phase_1_ue_ber_tmp)

        if cfg.csi_prediction:
            np.savez("results/channels_multiple_mu_mimo/results/{}/mu_mimo_results_UE_{}_pred.npz".format(folder_name, rx_ues_arr[ue_arr_idx]),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_bd[5],
                    nodewise_throughput=rst_bd[6], nodewise_bitrate=rst_bd[7], ranks=rst_bd[8], uncoded_ber_list=rst_bd[9],
                    ldpc_ber_list=rst_bd[10], sinr_dB=rst_bd[11])
        else:
            np.savez("results/channels_multiple_mu_mimo/results/{}/mu_mimo_results_UE_{}.npz".format(folder_name, rx_ues_arr[ue_arr_idx]),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_bd[5],
                    nodewise_throughput=rst_bd[6], nodewise_bitrate=rst_bd[7], ranks=rst_bd[8], uncoded_ber_list=rst_bd[9],
                    ldpc_ber_list=rst_bd[10], sinr_dB=rst_bd[11])