"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat


from dmimo.config import SimConfig, Ns3Config
from dmimo.ncjt_phase_3 import sim_ncjt_phase_3_all

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

script_name = sys.argv[0]
arguments = sys.argv[1:]

print(f"Script Name: {script_name}")
print(f"Arguments: {arguments}")

if len(arguments) > 0:
    mobility = arguments[0]
    drop_idx = arguments[1]
    rx_ues_arr = arguments[2:]
    rx_ues_arr = np.array(rx_ues_arr, dtype=int)
    
    print("Current mobility: {} \n Current drop: {} \n".format(mobility, drop_idx))
    print("rx_ues_arr: ", rx_ues_arr)
    print("rx_ues_arr[0]: ", rx_ues_arr[0])


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 0           # feedback delay in number of subframe
    cfg.perfect_csi = False
    cfg.rank_adapt = False      # disable rank adaptation
    cfg.link_adapt = False      # disable link adaptation
    cfg.csi_prediction = False
    cfg.receiver = 'LMMSE'      # 'LMMSE', 'PIC', 'SIC'
    cfg.num_tx_streams = 4
    num_rx_ues = cfg.num_tx_streams // 2
    cfg.num_rx_ue_sel = num_rx_ues
    cfg.modulation_order = 2
    cfg.lmmse_chest = True
    cfg.fft_size = 512
    cfg.dc_null = False
    if arguments == []:
        mobility = 'high_mobility'
        drop_idx = '1'
        rx_ues_arr = [2]
    
    # NS3 Configs
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    ns3cfg.num_rxue_sel = cfg.num_rx_ue_sel
    cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel) * 2), (ns3cfg.num_rxue_sel, -1))

    # Initializing storage variables
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

    #############################################
    # Testing
    #############################################

    uncoded_ber, uncoded_ser, per_stream_ber_all = sim_ncjt_phase_3_all(cfg, ns3cfg)

    file_path = "results/phase_3/usrp_channels_perfect_synch/{}_receiver/UEs_{}_streams_{}_modulation_order_{}.npz".format(
                cfg.receiver, num_rx_ues, cfg.num_tx_streams, cfg.modulation_order)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    np.savez(file_path, uncoded_ber=uncoded_ber, uncoded_ser=uncoded_ser, per_stream_ber_all=per_stream_ber_all)
