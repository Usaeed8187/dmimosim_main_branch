"""
Simulation of MU-MIMO scenario with ns-3 channels

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
from dmimo.mu_mimo import sim_mu_mimo_all


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 70     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.rank_adapt = False      # disable rank adaptation
    cfg.link_adapt = False      # disable link adaptation
    cfg.cfo_sigma = 0.0         # in Hz
    cfg.sto_sigma = 0.0         # in nanosecond
    cfg.ns3_folder = "../ns3/channels_medium_mobility/"

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("../results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    for num_rx_antennas in [8, 10, 12]:
        # Test case 1:  no rank adaptation, assuming 2 antennas per UE and treating BS as two UEs
        cfg.num_tx_streams = num_rx_antennas
        cfg.num_rx_ue_sel = (num_rx_antennas - 4) // 2
        cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
        cfg.ue_ranks = [2]  # same rank for all UEs

        # Test case 2: manual rank 1 adaption, assuming 2 antennas per UE and treating BS as two UEs
        # cfg.num_tx_streams = num_rx_antennas // 2
        # cfg.num_rx_ue_sel = (num_rx_antennas - 4) // 2
        # cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
        # cfg.ue_ranks = [1]  # same rank for all UEs

        # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
        modulation_orders = [2, 4, 6]
        num_modulations = len(modulation_orders)
        ber = np.zeros((2, num_modulations))
        ldpc_ber = np.zeros((2, num_modulations))
        goodput = np.zeros((2, num_modulations))
        throughput = np.zeros((2, num_modulations))

        for k in range(num_modulations):
            cfg.modulation_order = modulation_orders[k]

            cfg.csi_prediction = True
            cfg.precoding_method = "BD"
            rst_bd = sim_mu_mimo_all(cfg)
            ber[0, k] = rst_bd[0]
            ldpc_ber[0, k] = rst_bd[1]
            goodput[0, k] = rst_bd[2]
            throughput[0, k] = rst_bd[3]

            cfg.csi_prediction = True
            cfg.precoding_method = "ZF"
            rst_zf = sim_mu_mimo_all(cfg)
            ber[1, k] = rst_zf[0]
            ldpc_ber[1, k] = rst_zf[1]
            goodput[1, k] = rst_zf[2]
            throughput[1, k] = rst_zf[3]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        ax[0].set_title("MU-MIMO")
        ax[0].set_xlabel('Modulation (bits/symbol)')
        ax[0].set_ylabel('BER')
        ax[0].plot(modulation_orders, ber.transpose(), 'o-')
        ax[0].legend(['BD', 'ZF'])

        ax[1].set_title("MU-MIMO")
        ax[1].set_xlabel('Modulation (bits/symbol)')
        ax[1].set_ylabel('Coded BER')
        ax[1].plot(modulation_orders, ldpc_ber.transpose(), 'd-')
        ax[1].legend(['BD', 'ZF'])

        ax[2].set_title("MU-MIMO")
        ax[2].set_xlabel('Modulation (bits/symbol)')
        ax[2].set_ylabel('Goodput/Throughput (Mbps)')
        ax[2].plot(modulation_orders, goodput.transpose(), 's-')
        ax[2].plot(modulation_orders, throughput.transpose(), 'd-')
        ax[2].legend(['Goodput-BD', 'Goodput-ZF', 'Throughput-BD', 'Throughput-ZF'])

        basename = "../results/{}/mu_mimo_results_chanpred_s{}r{}".format(folder_name,
                                                                          cfg.num_tx_streams, cfg.ue_ranks[0])
        plt.savefig(f"{basename}.png")
        np.savez(f"{basename}.npz", ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput)
