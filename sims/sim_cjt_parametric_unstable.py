'''
This file lets you run the dMIMO CJT simulation by giving it the dMIMO configuration parameters such as 
the channel directory, output file name directory, GPU index to run on, and many more.

NOTE: This file is unstable: 
DO NOT pass "su-mimo" as dmimo_mode 
DO NOT enable scheduling by passing --enable_scheduling

Run python sims/sim_cjt_parametric_(unstable).py -h to see the list of parameters you can pass as arguments.

Example usage:
python sims/sim_cjt_parametric_(unstable).py 
--chan_dir ~/Projects/dMIMO/temp/dmimosim/ns3/channels/LowMobility 
--output_file_name ~/Projects/dMIMO/temp/dmimosim/results/new_folder/my_test_output.npz 
--dmimo_mode "mu-mimo"  
--total_slots 19 
--quick_run
'''


import argparse


class Args: # Define this solely for the intellisense to realize the attributes
    chan_dir: str                   # Channel directory
    output_file_name: str           # Output npz file path to store the results
    dmimo_mode: str                 # either "su-mimo" or "mu-mimo" 
    total_slots: int                # Ending index of the subframes to simulate (Will be overridden by --quick_run if passed)
    start_slot_idx: int             # Starting index of the subframes to simulate
    csi_delay: int                  # To determine the CSI feedback delay
    gpu_index: int                  # The GPU to use for simulation
    num_rx_ue:int                   # Number of Rx UEs participating
    enable_perfect_csi: bool        # Enables perfect channel knowledge on both transmitter and receiver
    disable_rank_link_adapt: bool   # Disables the rank and link adaptation
    enable_csi_prediction: bool     # Disables ESN-based CSI prediction at the transmitter
    enable_scheduling:bool          # Enables scheduling (not stable, do not pass)
    quick_run:bool                  # Whether to run only one subframe or not
    num_su_mimo_streams: int        # Number of streams to use in SU-MIMO mode
    modulation_order:int            # The number of bits per symbol for the modulation.


parser = argparse.ArgumentParser(description="Parse parameters for MIMO channel processing.")

# Required string arguments
parser.add_argument('--chan_dir', type=str, required=True, help="Directory containing channel files (required).")
parser.add_argument('--output_file_name', type=str, required=True, help="Output .npz file to save results (required).")
parser.add_argument('--dmimo_mode', type=str, required=True, choices=["su-mimo", "mu-mimo"],
                    help="MIMO mode: 'su-mimo' or 'mu-mimo' (required).")

# Optional integer arguments with defaults
parser.add_argument('--total_slots', type=int, default=90, help="Total number of slots in ns-3 channels (default: 90).")
parser.add_argument('--start_slot_idx', type=int, default=15,
                    help="Starting slot index (must be greater than csi_delay + 5) (default: 15).")
parser.add_argument('--csi_delay', type=int, default=4, help="Feedback delay in subframes (default: 4).")
parser.add_argument('--num_rx_ue', type=int, default=2, help="Number of UEs to be selected at the recieve squad (default: 2).")

# Optional boolean arguments with defaults
parser.add_argument('--enable_perfect_csi', action='store_true', default=False,
                    help="Use perfect CSI at both receiver and transmitter (default: False).")
parser.add_argument('--disable_rank_link_adapt', action='store_true', default=False,
                    help="Enable rank and link adaptation (default: False which means enabled).")
parser.add_argument('--enable_csi_prediction', action='store_true', default=False,
                    help="Enable WESN-based CSI prediction (default: False).")
parser.add_argument('--enable_scheduling', action='store_true', default=False,
                    help="Whether to enable UE scheduling (not stable, don't enable).")
parser.add_argument('--quick_run', action='store_true', default=False,
                    help="Pass this argument if you want to do a quick test run of just one subframe (default: False).")

parser.add_argument('--gpu_index', type=int, default=0, help="GPU index to use (default: 0).")

parser.add_argument('--modulation_order', type=int, default=2, help="Number of bits per symbol to use for modulation if rank and link adaptation is disabled (default: 2).")
parser.add_argument('--num_su_mimo_streams', type=int, default=2, help="Number of streams to use for SU-MIMO if rank and link adaptation is disabled (default: 2).")

# Parse the arguments
args:Args = parser.parse_args()

# Print parsed arguments (for debugging or verification)
print(args)



import sys
import os
# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

gpu_num = args.gpu_index  # Use "" to use the CPU, Use 0 to select first GPU
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

from dmimo.config import SimConfig, Ns3Config



# Add system path for the dmimo library
dmimo_root = os.path.abspath(os.path.dirname(__file__) + "/..")
sys.path.append(dmimo_root)

sys.path.append(os.path.join('..'))

# source_dir = '/home/data/ns3_channels_q4/'
# destination_dir = 'ns3/'
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)
# for root, dirs, files in os.walk(source_dir):
#     # Construct the relative path to replicate the directory structure
#     relative_path = os.path.relpath(root, source_dir)
#     destination_subdir = os.path.join(destination_dir, relative_path)

#     # Create the subdirectory in the destination if it doesn't exist
#     if not os.path.exists(destination_subdir):
#         os.makedirs(destination_subdir)
    
#     # Create symlinks for each file in the current directory
#     for file in files:
#         source_file = os.path.join(root, file)
#         destination_file = os.path.join(destination_subdir, file)

#         # If the symlink already exists, remove it
#         if os.path.exists(destination_file):
#             os.remove(destination_file)

#         # Create the symlink
#         os.symlink(source_file, destination_file)
#         # print(f"Symlink created for {source_file} -> {destination_file}")


# Main function
if __name__ == "__main__":


    mode = args.dmimo_mode
    if args.quick_run:
        args.total_slots = args.start_slot_idx + 1
    # Simulation settings
    cfg = SimConfig()
    cfg.total_slots = args.total_slots        # total number of slots in ns-3 channels
    cfg.start_slot_idx = args.start_slot_idx     # starting slots (must be greater than csi_delay + 5)
    cfg.csi_delay = args.csi_delay           # feedback delay in number of subframe
    cfg.perfect_csi = args.enable_perfect_csi
    cfg.rank_adapt = not (args.disable_rank_link_adapt)      # disable rank adaptation
    cfg.link_adapt = not (args.disable_rank_link_adapt)      # disable link adaptation
    cfg.csi_prediction = args.enable_csi_prediction
    mobility = 'medium_mobility'
    drop_idx = '3'
    # cfg.ns3_folder = "ns3/channels_" + mobility + '_' + drop_idx + '/'
    # "ns3/channels/LowMobility"
    cfg.ns3_folder = os.path.join(dmimo_root, os.path.relpath(args.chan_dir))
    ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)

    folder_name = os.path.basename(os.path.abspath(cfg.ns3_folder))
    os.makedirs(os.path.join("results", folder_name), exist_ok=True)
    print("Using channels in {}".format(folder_name))    

    # rx_ues_arr = [1,2,4,6]
    rx_ues_arr = [args.num_rx_ue]    

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
    phase_1_ue_ber = []

    if mode == 'mu-mimo':
        from dmimo.mu_mimo_integration import sim_mu_mimo_all
        

        for ue_arr_idx in range(np.size(rx_ues_arr)):
            cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
            ns3cfg.num_rxue_sel = cfg.num_rx_ue_sel

            if args.disable_rank_link_adapt:
                
                num_rx_antennas = rx_ues_arr[ue_arr_idx] * 2 + 4

                # Test case 1:  rank 2 transmission, assuming 2 antennas per UE and treating BS as two UEs
                # cfg.num_tx_streams = num_rx_antennas
                # cfg.ue_ranks = [2] # same rank for all UEs

                # Test case 2: rank 1 transmission, assuming 2 antennas per UE and treating BS as two UEs
                cfg.num_tx_streams = num_rx_antennas // 2
                cfg.ue_ranks = [1]  # same rank for all UEs

                # Modulation order: 2/4/6 for QPSK/16QAM/64QAM
                cfg.modulation_order = 4
            
            if args.enable_scheduling:
                tx_ue_mask = np.ones(cfg.num_tx_ue_sel)
                rx_ue_mask = np.ones(cfg.num_rx_ue_sel)
                ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

            cfg.ue_indices = np.reshape(np.arange((ns3cfg.num_rxue_sel + 2) * 2), (ns3cfg.num_rxue_sel + 2, -1))

            cfg.precoding_method = "ZF"
            rst_zf = sim_mu_mimo_all(cfg, ns3cfg)
            ber[ue_arr_idx] = rst_zf[0]
            ldpc_ber[ue_arr_idx] = rst_zf[1]
            goodput[ue_arr_idx] = rst_zf[2]
            throughput[ue_arr_idx] = rst_zf[3]
            bitrate[ue_arr_idx] = rst_zf[4]
            
            nodewise_goodput.append(rst_zf[5])
            nodewise_throughput.append(rst_zf[6])
            nodewise_bitrate.append(rst_zf[7])
            ranks.append(rst_zf[8])
            uncoded_ber_list.append(rst_zf[9])
            ldpc_ber_list.append(rst_zf[10])
            if rst_zf[11] is not None:
                sinr_dB.append(np.concatenate(rst_zf[11]))

            folder_path = "results/channels_multiple_mu_mimo/{}".format(folder_name)
            os.makedirs(folder_path, exist_ok=True)

            file_path = args.output_file_name
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            np.savez(file_path,
                    ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, nodewise_goodput=rst_zf[5],
                    nodewise_throughput=rst_zf[6], nodewise_bitrate=rst_zf[7], ranks=rst_zf[8], uncoded_ber_list=rst_zf[9],
                    ldpc_ber_list=rst_zf[10], sinr_dB=rst_zf[11])

            
                    
    elif mode == 'su-mimo':
        from dmimo.su_mimo_kpi import sim_su_mimo_all
        if not args.disable_rank_link_adapt:
            for ue_arr_idx in range(np.size(rx_ues_arr)):

                cfg.num_rx_ue_sel = rx_ues_arr[ue_arr_idx]
                cfg.ue_indices = np.reshape(np.arange((cfg.num_rx_ue_sel + 2) * 2), (cfg.num_rx_ue_sel + 2, -1))
                    
                cfg.precoding_method = "ZF"
                rst_zf = sim_su_mimo_all(cfg)
                ber = rst_zf[0]
                ldpc_ber = rst_zf[1]
                goodput = rst_zf[2]
                throughput = rst_zf[3]
                bitrate = rst_zf[4]

                ranks = rst_zf[5]
                ldpc_ber_list = rst_zf[6]
                uncoded_ber_list = rst_zf[7]

                if cfg.csi_prediction:
                    np.savez(
                        # "results/{}/su_mimo_results_UE_{}_pred.npz".format(folder_name),
                                args.output_file_name,
                                ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                                ldpc_ber_list=ldpc_ber_list)
                else:
                    np.savez(
                        # "results/{}/su_mimo_results_UE_{}.npz".format(folder_name, rx_ues_arr[ue_arr_idx]),
                                args.output_file_name,
                                ber=ber, ldpc_ber=ldpc_ber, goodput=goodput, throughput=throughput, bitrate=bitrate, ranks=ranks, uncoded_ber_list=uncoded_ber_list,
                                ldpc_ber_list=ldpc_ber_list)
        else:
            cfg.rank_adapt = False
            cfg.link_adapt = False
            
            # Test 1 parameters
            cfg.num_tx_streams = args.num_su_mimo_streams
            cfg.modulation_order = args.modulation_order
            cfg.code_rate = 0.5

            cfg.precoding_method = "ZF"
            rst_svd = sim_su_mimo_all(cfg)
            ber[1] = rst_svd[0]
            ldpc_ber[1] = rst_svd[1]
            goodput[1] = rst_svd[2]
            throughput[1] = rst_svd[3]
            pass
    else:
        raise ValueError('dmimo-mode can only be "mu-mimo" or "su-mimo", but <',mode,'> was given.')