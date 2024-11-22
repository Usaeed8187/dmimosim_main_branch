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
from dmimo.generate_rx_squad_data import sim_mu_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 100        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 0     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 0           # feedback delay in number of subframe
    cfg.rank_adapt = False      # disable rank adaptation
    cfg.link_adapt = False      # disable link adaptation
    cfg.csi_prediction = False
    mobility = 'medium_mobility'
    drop_idx = '3'
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))
    
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    cfg.num_tx_streams = 4
    rx_ues_arr = 2

    cfg.num_rx_ue_sel = rx_ues_arr
    ns3cfg.num_rxue_sel = cfg.num_rx_ue_sel
    cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

    #############################################
    # Testing with rank and link adaptation
    #############################################

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        cfg.precoding_method = "ZF"
        x_rg_all_4_streams, x_rg_all_2_streams = sim_mu_mimo_all(cfg, ns3cfg)

        np.savez("results/data_2_and_4_streams/data_2_streams.npz", x_rg_all_2_streams=x_rg_all_2_streams)
        np.savez("results/data_2_and_4_streams/data_4_streams.npz", x_rg_all_4_streams=x_rg_all_4_streams)