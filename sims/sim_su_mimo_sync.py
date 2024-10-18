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
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

from dmimo.config import SimConfig
from dmimo.su_mimo import sim_su_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.csi_prediction = False  # use channel prediction or not
    cfg.total_slots = 95        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 8           # feedback delay in number of subframe
    cfg.rank_adapt = False      # disable rank adaptation
    cfg.link_adapt = False      # disable link adaptation
    cfg.gen_sync_errors = True  # random CFO/STO errors in each simulation cycle

    cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels_medium_mobility/")

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join(dmimo_root, "results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    cfg.num_tx_streams = 6      # total number of streams

    # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
    modulation_orders = [2, 4, 6]
    num_modulations = len(modulation_orders)
    ber = np.zeros((2, num_modulations))
    ldpc_ber = np.zeros((2, num_modulations))
    goodput = np.zeros((2, num_modulations))
    throughput = np.zeros((2, num_modulations))

    for sto in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
        for cfo in [0, 100, 200, 300, 400, 500, 600]:
            cfg.sto_sigma = sto
            cfg.cfo_sigma = cfo

            for k in range(num_modulations):
                cfg.modulation_order = modulation_orders[k]

                cfg.precoding_method = "SVD"
                rst_svd = sim_su_mimo_all(cfg)
                ber[0, k] = rst_svd[0]
                ldpc_ber[0, k] = rst_svd[1]
                goodput[0, k] = rst_svd[2]
                throughput[0, k] = rst_svd[3]

                cfg.precoding_method = "ZF"
                rst_zf = sim_su_mimo_all(cfg)
                ber[1, k] = rst_zf[0]
                ldpc_ber[1, k] = rst_zf[1]
                goodput[1, k] = rst_zf[2]
                throughput[1, k] = rst_zf[3]

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            ax[0].set_title("SU-MIMO")
            ax[0].set_xlabel('Modulation (bits/symbol)')
            ax[0].set_ylabel('BER')
            ax[0].plot(modulation_orders, ber.transpose(), 'o-')
            ax[0].legend(['SVD', 'ZF'])

            ax[1].set_title("SU-MIMO")
            ax[1].set_xlabel('Modulation (bits/symbol)')
            ax[1].set_ylabel('Coded BER')
            ax[1].plot(modulation_orders, ldpc_ber.transpose(), 'd-')
            ax[1].legend(['SVD', 'ZF'])

            ax[2].set_title("SU-MIMO")
            ax[2].set_xlabel('Modulation (bits/symbol)')
            ax[2].set_ylabel('Goodput/Throughput (Mbps)')
            ax[2].plot(modulation_orders, goodput.transpose(), 's-')
            ax[2].plot(modulation_orders, throughput.transpose(), 'd-')
            ax[2].legend(['Goodput-SVD', 'Goodput-ZF', 'Throughput-SVD', 'Throughput-ZF'])

            basename = dmimo_root + "/results/{}/su_mimo_results_cfo{}_sto{}".format(folder_name, cfo, sto)
            plt.savefig(f"{basename}.png")
            plt.close(fig)
            np.savez(f"{basename}.npz", ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)

