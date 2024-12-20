"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat


from dmimo.config import SimConfig, Ns3Config, MCSConfig
from dmimo.phase_3_PIC_demo import test_phase_3_rx_all

gpu_num = 0  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm/16/lib64/libLLVM.so'

# Configure to use only a single GPU and allocate only as much memory as needed
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

# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.perfect_csi = False
    cfg.rank_adapt = False      # disable rank adaptation
    cfg.link_adapt = False      # disable link adaptation
    cfg.csi_prediction = False
    cfg.receiver = 'PIC'      # 'LMMSE', 'PIC'
    cfg.num_tx_streams = 4
    mcsconfig = MCSConfig()
    rx_ues_arr = 2

    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    cfg.num_rx_ue_sel = rx_ues_arr
    ns3cfg.num_rxue_sel = cfg.num_rx_ue_sel
    cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

    #############################################
    # Loading Transmit and Receive Data
    #############################################

    rx_data = loadmat('tests/usrp_rx_sigs/phase_3/y_freq.mat')
    y = rx_data['y_freq']
    y_rg = y.transpose(2,3,1,0)[:, np.newaxis, ...]

    if cfg.num_tx_streams == 2:
        tx_data = np.load('tests/usrp_tx_sigs/phase_3/data_2_streams.npz')
        x_rg = tx_data['x_rg_all_2_streams']
    elif cfg.num_tx_streams == 4:
        tx_data = np.load('tests/usrp_tx_sigs/phase_3/data_4_streams.npz')
        x_rg = tx_data['x_rg_all_4_streams']
    else:
        raise ValueError('Number of streams should be 2 or 4')

    new_shape = [x_rg.shape[0] * x_rg.shape[1]] + list(x_rg.shape[2:])
    x_rg = tf.reshape(x_rg, new_shape)

    #############################################
    # Testing
    #############################################

    uncoded_ber, uncoded_ser = test_phase_3_rx_all(cfg, ns3cfg, x_rg, y_rg)

    file_path = "results/phase_3/usrp_channels_perfect_synch/{}_receiver/UEs_{}_streams_{}_modulation_order_{}.npz".format(
                cfg.receiver, rx_ues_arr, cfg.num_tx_streams, mcsconfig.modulation_order)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    np.savez(file_path, uncoded_ber=uncoded_ber, uncoded_ser=uncoded_ser)