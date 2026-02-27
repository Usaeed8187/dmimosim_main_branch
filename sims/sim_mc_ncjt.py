"""
Simulation of NCJT scenario with ns-3 channels

"""

import os
import sys
from fractions import Fraction
# import numpy as np
# import matplotlib.pyplot as plt

gpu_num = 1  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/llvm/16/lib64/libLLVM.so'
# os.environ['DRJIT_LIBLLVM_PATH'] = '/usr/lib/x86_64-linux-gnu/libLLVM-16.so'


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

from dmimo.config import SimConfig, Ns3Config
from dmimo.mc_ncjt import sim_mc_ncjt

def _parse_code_rate(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return float(value)


def parse_arguments(arguments):
    drop_idx = "5"
    rx_ues = 3
    modulation_order = 6
    code_rate = 1 / 2
    num_txue_sel = 10

    if len(arguments) >= 1:
        drop_idx = str(arguments[0])
    if len(arguments) >= 2:
        rx_ues = int(arguments[1])
    if len(arguments) >= 3:
        num_txue_sel = int(arguments[2])
    if len(arguments) >= 4:
        modulation_order = int(arguments[3])
    if len(arguments) >= 5:
        code_rate = _parse_code_rate(arguments[4])

    return drop_idx, rx_ues, num_txue_sel, modulation_order, code_rate

# Main function
if __name__ == "__main__":
    import sionna
    arguments = sys.argv[1:]
    drop_idx, rx_ues, num_txue_sel, modulation_order, code_rate = parse_arguments(arguments)
    # for channel_seed in range(2,7):
        # print(f"Running with channel random seed {channel_seed}")
    # Set sionna random seed
    sionna.config.seed = 26
    channel_seed = 4
    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = 26 #65        # total number of slots in ns-3 channels
    cfg.start_slot_idx = 16 #15     # starting slots (must be greater than csi_delay + 5)
    cfg.num_slots_p1 = 2
    cfg.num_slots_p2 = 2

    # cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels/HighMobilitySeed%d" % channel_seed)  # folder where the ns-3 channels are stored
    # cfg.ns3_folder = os.path.join(dmimo_root, "ns3/channels/LowMobility")  # folder where the ns-3 channels are stored
    cfg.ns3_folder = os.path.join(dmimo_root, f"ns3/channels_high_mobility_{drop_idx}")
    cfg.enable_ue_selection = False
    cfg.perfect_csi = False

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join(dmimo_root, "results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))

    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
    # ns3cfg.thermal_noise = -180
    # Select different number of Tx/Rx nodes

    # We set the antennas of each cluster such that the total transmit power is the same between transmit clusters.
    ns3cfg.num_txue_sel = num_txue_sel
    # cluster_ant_list = [list(range(4)), list(range(4,24))]
    cluster_ant_list = [[0,1] + list(range(4,4+ns3cfg.num_txue_sel//2*2)) , [2,3] + list(range(4+ns3cfg.num_txue_sel//2*2,4+ns3cfg.num_txue_sel*2))]
    # cluster_ant_list = [[0,1,2,4], list(range())]
    modulation_order = modulation_order
    mod_order_list = [modulation_order, modulation_order] # [4,4] # Modulation order cluster 1 and 2
    ns3cfg.num_rxue_sel = rx_ues

    # cfg.modulation_order = 4
    cfg.code_rate = code_rate

    cfg.enable_ue_selection = False

    # Tx Squad settings
    cfg.phase_1_enabled = True
    # Rx Squad settings
    cfg.ncjt_ldpc_decode_and_forward = True
    cfg.enable_rxsquad = True

    # Run the simulation
    # benchmark the simulation time
    import time
    start_time = time.time()
    uncoded_ber, coded_ber, coded_bler, goodput, throughput = sim_mc_ncjt(cfg, ns3cfg, cluster_ant_list, mod_order_list, RB_based_ue_selection=False, num_selected_ues = 10 , perSC_SNR=False)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds ({(end_time - start_time)/(cfg.total_slots - cfg.start_slot_idx)} seconds per slot)")
    # Show results
    print(f"Average uncoded/coded BER: {uncoded_ber}  {coded_ber}")
    print(f"Average coded BLER: {coded_bler}")
    print(f"Average goodput/throughput: {goodput}  {throughput}")

