"""
Simulation of NCJT scenario with ns-3 channels

This scripts should be called from the "sims" folder
"""

# add system folder for the dmimo library
import os
import sys

current_dir = (os.getcwd())
indx = current_dir.find('dmimosim')
sys.path.append(current_dir[:indx + len('dmimosim')])

# Comment this part out if it doesn't matter which GPU to use:
gpu_num = '0'  # Use "" to use the CPU, Use 0 to select first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np

from dmimo.config.ncjtsimconfig import NcjtSimConfig
from dmimo.ncjt.ncjt_old import ncjt_sim_all_phases


# Main function
if __name__ == "__main__":

    # Simulation settings
    cfg = NcjtSimConfig()
    cfg.num_subframes = 9  # total number of subframes in ns-3 channels
    cfg.num_ofdm_symbols = 14  # number of OFDM symbols per subframe
    cfg.num_bits_per_symbol_phase2 = 2  # 2 for QPSK and 4 for 16QAM
    cfg.ns3_folder = '../ns3/channels'
    cfg.perSC_SNR = False
    # cfg.NOISE_FLOOR = -180  # uncomment if you want to have zero noise.
    cfg.num_bits_per_symbol_phase1 = 8
    cfg.num_bits_per_symbol_phase2 = 4
    cfg.num_subframes_phase1 = 3
    cfg.num_subframes_phase2 = 6

    assert cfg.num_bits_per_symbol_phase2 * cfg.num_subframes_phase2 \
           == cfg.num_bits_per_symbol_phase1 * cfg.num_subframes_phase1

    # How many simulations with the same setup to average out the effect of random AWGN and information bits.
    num_trials = 20  # FIXME should not do this

    num_TxUes_list = [0, 2, 4, 6, 8, 10]
    num_RxUes_list = [0, 2, 4, 6, 8, 10]

    BER_list = np.zeros((len(num_TxUes_list), len(num_RxUes_list), num_trials))
    for i_num_TxUes, num_TxUes in enumerate(num_TxUes_list):
        cfg.num_TxUe = num_TxUes
        for i_num_RxUes, num_RxUes in enumerate(num_RxUes_list):
            cfg.num_RxUe = num_RxUes
            print(f'Simulating {num_TxUes} Tx UEs and {num_RxUes} Rx UEs')

            for i_trial in range(num_trials):
                avg_ber = ncjt_sim_all_phases(cfg)
                BER_list[i_num_TxUes, i_num_RxUes, i_trial] = avg_ber
            pass
    BER_list = np.mean(BER_list, axis=-1)

    folder = '../results/NCJT_Alamouti_QTR3/'
    child = 'QPSK/' if cfg.num_bits_per_symbol_phase2 == 2 else f'{2 ** cfg.num_bits_per_symbol_phase2}QAM/'
    child = child + ('ExactSNR/' if cfg.perSC_SNR else 'noSNR/')
    os.makedirs(folder + child, exist_ok=True)
    np.save(folder + child + f'BER_list.npy', BER_list)
    np.save(folder + child + f'num_TxUes_list.npy', num_TxUes_list)
    np.save(folder + child + f'num_RxUes_list.npy', num_RxUes_list)

    print('Done')
