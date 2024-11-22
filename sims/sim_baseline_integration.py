"""
Simulation of baseline scenario with ns-3 channels

This scripts should be called from the "tests" folder
"""

# add system folder for the dmimo library
import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np

from dmimo.config import SimConfig, Ns3Config
from dmimo.baseline_integration import sim_baseline_all

gpu_num = 0  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    cfg.total_slots = 100        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 2           # feedback delay in number of subframe
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    mobility = 'medium_mobility'
    drop_idx = '3'
    cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.rank_adapt = True
    cfg.link_adapt = True
    cfg.csi_prediction = False

    cfg.precoding_method = "ZF"
    rst_zf = sim_baseline_all(cfg, ns3cfg)
    ber = rst_zf[0]
    ldpc_ber = rst_zf[1]
    goodput = rst_zf[2]
    throughput = rst_zf[3]
    bitrate = rst_zf[4]

    ranks = rst_zf[5]
    ldpc_ber_list = rst_zf[6]
    uncoded_ber_list = rst_zf[7]
    
    if cfg.csi_prediction:
        np.savez("results/{}/baseline_results_pred.npz".format(folder_name),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                    ldpc_ber_list=ldpc_ber_list)
    else:
        np.savez("results/{}/baseline_results.npz".format(folder_name),
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                    ldpc_ber_list=ldpc_ber_list)

    #############################################
    # Testing without rank and link adaptation
    #############################################

    # cfg.rank_adapt = False
    # cfg.link_adapt = False
    # cfg.csi_prediction = False

    # cfg.num_tx_streams = 4
    # cfg.modulation_order = 2
    # cfg.code_rate = 0.5

    # cfg.precoding_method = "ZF"
    # rst_zf = sim_baseline_all(cfg, ns3cfg)
    # ber = rst_zf[0]
    # ldpc_ber = rst_zf[1]
    # goodput = rst_zf[2]
    # throughput = rst_zf[3]
    # bitrate = rst_zf[4]

    # ranks = rst_zf[5]
    # ldpc_ber_list = rst_zf[6]
    # uncoded_ber_list = rst_zf[7]
    
    # if cfg.csi_prediction:
    #     np.savez("results/{}/baseline_results_pred.npz".format(folder_name),
    #                 ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
    #                 ldpc_ber_list=ldpc_ber_list)
    # else:
    #     np.savez("results/{}/baseline_results.npz".format(folder_name),
    #                 ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
    #                 ldpc_ber_list=ldpc_ber_list)