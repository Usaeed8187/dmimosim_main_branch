import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode


#############################################
# Settings
#############################################

rx_ues_arr = [2]

precoding_method = 'weighted_mean'
mobility = 'high_mobility'
drops = np.arange(1,9)
modulation_order = 4

num_tx_streams = 1

SNR_range = np.arange(0, 40, 2)

#############################################
# KPI Handling
#############################################

averaging_method_ber = []
weighted_averaging_method_ber = []


for num_scheduled_tx_ue in rx_ues_arr:

    plt.figure()

    uncoded_bers = []

    for drop_idx in drops:
        file_path = "results/phase_1_sim_dl_channels/{}_precoding_method/{}_drop_idx_{}_UEs_{}_streams_per_tx_{}_modulation_order_{}.npz".format(
                        precoding_method, mobility, drop_idx, num_scheduled_tx_ue, num_tx_streams, modulation_order)
        data = np.load(file_path)

        uncoded_bers.append(data['uncoded_bers'])

    uncoded_bers = np.asarray(uncoded_bers)
    uncoded_bers = np.mean(uncoded_bers, axis=(0,1))

    worst_users = np.argmax(uncoded_bers, axis=-1)
    worst_users  = mode(worst_users, axis=1, keepdims=False).mode

    plt.semilogy(SNR_range, uncoded_bers[0,:,worst_users[0]], label='Worst user (user {}) after mean precoding'.format(worst_users[0]))
    plt.semilogy(SNR_range, uncoded_bers[1,:,worst_users[1]], label='Worst user (user {}) after weighted mean precoding'.format(worst_users[1]))

    plt.legend()
    plt.grid()
    plt.xlabel('Best User SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs SNR for {} Total TX UEs with 5 dB difference in SNR (16 QAM, {} Streams)'.format(num_scheduled_tx_ue, num_tx_streams))

    plt.savefig('Mean and Weighted Mean - {} Users'.format(num_scheduled_tx_ue))