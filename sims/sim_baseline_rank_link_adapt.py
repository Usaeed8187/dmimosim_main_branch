"""
Simulation of Baseline scenario with ns-3 channels

This scripts should be called from the "sims" folder
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
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# add system folder for the dmimo library
sys.path.append(os.path.join('..'))

from dmimo.config import SimConfig
from dmimo.baseline import sim_baseline_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 35        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 2           # feedback delay in number of subframe
    cfg.ns3_folder = "../ns3/channels_medium_mobility/"

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    ber = np.zeros(3)
    ldpc_ber = np.zeros(3)
    goodput = np.zeros(3)
    throughput = np.zeros(3)

    #############################################
    # Testing with rank and link adaptation
    #############################################

    cfg.rank_adapt = True
    cfg.link_adapt = True

    cfg.precoding_method = "ZF"
    rst_svd = sim_baseline_all(cfg)
    ber[0] = rst_svd[0]
    ldpc_ber[0] = rst_svd[1]
    goodput[0] = rst_svd[2]
    throughput[0] = rst_svd[3]

    #############################################
    # Testing without rank and link adaptation
    #############################################

    cfg.rank_adapt = False
    cfg.link_adapt = False

    # Test 1 parameters
    cfg.num_tx_streams = 2
    cfg.modulation_order = 2
    cfg.code_rate = 0.5

    cfg.precoding_method = "SVD"
    rst_svd = sim_baseline_all(cfg)
    ber[1] = rst_svd[0]
    ldpc_ber[1] = rst_svd[1]
    goodput[1] = rst_svd[2]
    throughput[1] = rst_svd[3]

    # Test 2 parameters
    cfg.num_tx_streams = 4
    cfg.modulation_order = 4
    cfg.code_rate = 0.5

    cfg.precoding_method = "SVD"
    rst_svd = sim_baseline_all(cfg)
    ber[2] = rst_svd[0]
    ldpc_ber[2] = rst_svd[1]
    goodput[2] = rst_svd[2]
    throughput[2] = rst_svd[3]

    np.savez("../results/{}/baseline_results_adapt.npz".format(folder_name),
             ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)
