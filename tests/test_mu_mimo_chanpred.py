"""
Simulation of MU-MIMO scenario with ns-3 channels

"""

import sys
import os
import numpy as np
# import matplotlib.pyplot as plt

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
from dmimo.mu_mimo import sim_mu_mimo


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 90        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 70     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = 4           # feedback delay in number of subframe
    cfg.num_tx_streams = 8      # 4/6/8/12 equal to total number of streams
    cfg.link_adapt = False      # disable link adaptation
    cfg.rank_adapt = False      # disable rank adaptation

    cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels_high_mobility/")
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    avg_ber, avg_ber_pred = 0.0, 0.0
    avg_goodput, avg_tput = 0.0, 0.0
    avg_goodput_pred, avg_tput_pred = 0.0, 0.0
    total_runs = 0
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1+cfg.num_slots_p2):
        total_runs += 1
        print("------ Run {} -----".format(total_runs))
        cfg.first_slot_idx = first_slot_idx
        cfg.csi_prediction = False
        cfg.precoding_method = "ZF"
        bers, bits = sim_mu_mimo(cfg, ns3cfg)
        avg_ber += bers[0]
        avg_goodput += bits[0]
        avg_tput += bits[1]
        print("Channel prediction: ", cfg.csi_prediction)
        print("BER: ", bers)
        print("Goodbits: ", bits)

        cfg.csi_prediction = True
        cfg.precoding_method = "ZF"
        bers, bits = sim_mu_mimo(cfg, ns3cfg)
        avg_ber_pred += bers[0]
        avg_goodput_pred += bits[0]
        avg_tput_pred += bits[1]
        print("Channel prediction: ", cfg.csi_prediction)
        print("BER: ", bers)
        print("Goodbits: ", bits)

    avg_ber /= total_runs
    avg_ber_pred /= total_runs

    scaling = cfg.num_slots_p2 / (cfg.num_slots_p1 + cfg.num_slots_p2) / (cfg.slot_duration * 1e6)
    avg_goodput = avg_goodput / total_runs * scaling
    avg_goodput_pred = avg_goodput_pred / total_runs * scaling
    avg_tput = avg_tput / total_runs * scaling
    avg_tput_pred = avg_tput_pred / total_runs * scaling

    print("")
    print("Average BER: {:3f}".format(avg_ber))
    print("Average Goodput: {:.2f}, Average Throughput: {:.2f}".format(avg_goodput, avg_tput))
    print("Average BER with prediction: {:3f}".format(avg_ber_pred))
    print("Average Goodput: {:.2f}, Average Throughput: {:.2f}".format(avg_goodput_pred, avg_tput_pred))

