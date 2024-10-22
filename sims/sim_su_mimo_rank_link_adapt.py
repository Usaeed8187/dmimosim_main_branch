"""
Simulation of SU-MIMO scenario with ns-3 channels

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

from dmimo.config import SimConfig, Ns3Config
from dmimo.su_mimo import sim_su_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 35        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 8           # feedback delay in number of subframe
    cfg.rank_adapt = True       # disable rank adaptation
    cfg.link_adapt = True       # disable link adaptation

    cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels_medium_mobility/")
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join(dmimo_root, "results", folder_name), exist_ok=True)
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
    rst_svd = sim_su_mimo_all(cfg, ns3cfg)
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

    cfg.precoding_method = "ZF"
    rst_svd = sim_su_mimo_all(cfg, ns3cfg)
    ber[1] = rst_svd[0]
    ldpc_ber[1] = rst_svd[1]
    goodput[1] = rst_svd[2]
    throughput[1] = rst_svd[3]

    # Test 2 parameters
    cfg.num_tx_streams = 6
    cfg.modulation_order = 6
    cfg.code_rate = 0.5

    cfg.precoding_method = "ZF"
    rst_svd = sim_su_mimo_all(cfg, ns3cfg)
    ber[2] = rst_svd[0]
    ldpc_ber[2] = rst_svd[1]
    goodput[2] = rst_svd[2]
    throughput[2] = rst_svd[3]

    basename = dmimo_root + "/results/{}/su_mimo_results_adapt.npz".format(folder_name)
    plt.savefig(f"{basename}.png")
    np.savez(f"{basename}.npz", ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)
