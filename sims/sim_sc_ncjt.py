"""
Simulation of NCJT scenario with ns-3 channels

"""

import os
import sys
# import numpy as np
# import matplotlib.pyplot as plt

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

# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

from dmimo.config import SimConfig
from dmimo.sc_ncjt import sim_sc_ncjt


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 65        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 15     # starting slots (must be greater than csi_delay + 5)
    cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels_medium_mobility/")

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join(dmimo_root, "results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    # Select different number of Tx/Rx nodes
    cfg.num_tx_ue_sel = 8
    cfg.num_rx_ue_sel = 6
    cfg.modulation_order = 4
    cfg.code_rate = 0.5

    # Run the simulation
    uncoded_ber, coded_ber, coded_bler, goodput, throughput = sim_sc_ncjt(cfg)

    # Show results
    print(f"Average uncoded/coded BER: {uncoded_ber}  {coded_ber}")
    print(f"Average coded BLER: {coded_bler}")
    print(f"Average goodput/throughput: {goodput}  {throughput}")

