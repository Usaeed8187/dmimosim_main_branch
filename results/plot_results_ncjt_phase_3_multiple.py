import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np
import math



#############################################
# Settings
#############################################

rx_ues_arr = [4]
scheduling = False
prediction_results = False
if scheduling:
    rx_ues_arr = [10]

# Define positions for the box plots
positions_all_scenarios = []
positions_scenario_1 = np.arange(1,np.size(rx_ues_arr)+1)
positions_all_scenarios.append(positions_scenario_1)
positions_all_scenarios.append(positions_scenario_1 + 5) # Shifted right for the second scenario
positions_all_scenarios.append(positions_scenario_1 + 10) # Shifted further right for the third scenario

mobilities = ['high_mobility']

receiver = 'SIC' # 'SIC', 'LMMSE'
precoding = 'eigenmode' # 'eigenmode', 'none'

streams_per_tx = 1
modulation_order = 4

drop_arr = np.arange(1,9)
# drop_arr = np.asarray([1])

sto_arr = [0, 10, 30, 50, 70]
cfo_arr = [0, 100, 250, 400, 550]

sto = sto_arr[4]
cfo = cfo_arr[0]

gen_sync_errors = True

#############################################
# KPI Handling
#############################################

def return_KPIs(rx_ues_arr, scheduling, mobilities, receiver, precoding, streams_per_tx, modulation_order, drop_arr, sto, cfo):
    uncoded_ber = []
    ldpc_ber = []
    throughput = []

    for mobility_idx in range(np.size(mobilities)):

        curr_mobility = mobilities[mobility_idx]

        temp_uncoded_ber = []
        temp_ldpc_ber = []
        temp_throughput = []

        for ue_arr_idx in range(np.size(rx_ues_arr)):

            uncoded_ber_all_drops = []
            ldpc_ber_all_drops = []
            throughput_all_drops = []

            for drop_idx in drop_arr:
                
                if gen_sync_errors:
                    folder_path = "results/phase_3_sim_dl_channels_gen_sync_errors_True/{}_receiver_{}_precoding_method/{}_drop_idx_{}".format(receiver, precoding, curr_mobility, drop_idx)

                    if scheduling:
                        file_path = folder_path + "_scheduling_streams_per_tx_{}_modulation_order_{}_cfo_{}_sto_{}.npz".format(streams_per_tx, modulation_order, cfo, sto)
                    else:
                        file_path = folder_path + "_UEs_{}_streams_per_tx_{}_modulation_order_{}_cfo_{}_sto_{}.npz".format(rx_ues_arr[ue_arr_idx], streams_per_tx, modulation_order, cfo, sto)
                    if prediction_results:
                        file_path = folder_path + "_UEs_{}_streams_per_tx_{}_modulation_order_{}_cfo_{}_sto_{}_prediction.npz".format(rx_ues_arr[ue_arr_idx], streams_per_tx, modulation_order, cfo, sto)

                else:
                    folder_path = "results/phase_3_sim_dl_channels_gen_sync_errors_False/{}_receiver_{}_precoding_method/{}_drop_idx_{}".format(receiver, precoding, curr_mobility, drop_idx)
                    
                    if scheduling:
                        file_path = folder_path + "_scheduling_streams_per_tx_{}_modulation_order_{}.npz".format(streams_per_tx, modulation_order)
                    else:
                        file_path = folder_path + "_UEs_{}_streams_per_tx_{}_modulation_order_{}.npz".format(rx_ues_arr[ue_arr_idx], streams_per_tx, modulation_order)
                    if prediction_results:
                        file_path = folder_path + "_UEs_{}_streams_per_tx_{}_modulation_order_{}_prediction.npz".format(rx_ues_arr[ue_arr_idx], streams_per_tx, modulation_order)
                
                try:
                    data = np.load(file_path)

                    uncoded_ber_all_drops.append(data['uncoded_ber'])
                    ldpc_ber_all_drops.append(data['ldpc_ber'])
                    throughput_all_drops.append(data['throughput'])
                except:
                    continue

                # print("Using drop: ", drop_idx)
            
            if len(uncoded_ber_all_drops) > 1:
                temp_uncoded_ber.append(np.concatenate([uncoded_ber_all_drops]))
                temp_ldpc_ber.append(np.concatenate([ldpc_ber_all_drops]))
                temp_throughput.append(np.concatenate([throughput_all_drops]))
            else:
                temp_uncoded_ber.append(uncoded_ber_all_drops)
                temp_ldpc_ber.append(ldpc_ber_all_drops)
                temp_throughput.append(throughput_all_drops)

        uncoded_ber.append(temp_uncoded_ber)
        ldpc_ber.append(temp_ldpc_ber)
        throughput.append(temp_throughput)

    #############################################
    # Plots
    #############################################

    print('\nResults for {} precoding method and {} receiver'.format(precoding, receiver))
    uncoded_ber = np.mean(np.asarray(uncoded_ber), axis=-1)
    ldpc_ber = np.mean(np.asarray(ldpc_ber), axis=-1)
    throughput = np.mean(np.asarray(throughput), axis=-1)
    print("uncoded ber = ", np.asarray(uncoded_ber))
    print("ldpc ber = ", np.asarray(ldpc_ber))
    print("throughput = ", np.asarray(throughput))

    return [uncoded_ber, ldpc_ber, throughput]

uncoded_ber = np.zeros((len(cfo_arr), len(sto_arr)))
ldpc_ber = np.zeros((len(cfo_arr), len(sto_arr)))
throughput = np.zeros((len(cfo_arr), len(sto_arr)))
for cfo_idx, cfo in enumerate(cfo_arr):
    for sto_idx, sto in enumerate(sto_arr):
        kpis = return_KPIs(rx_ues_arr, scheduling, mobilities, receiver, precoding, streams_per_tx, modulation_order, drop_arr, sto, cfo)

        uncoded_ber[cfo_idx, sto_idx] = kpis[0].flatten()[0]
        ldpc_ber[cfo_idx, sto_idx] = kpis[1].flatten()[0]
        throughput[cfo_idx, sto_idx] = kpis[2].flatten()[0]

uncoded_ber_cfos = uncoded_ber[:,0]
uncoded_ber_stos = uncoded_ber[0,:]

ldpc_ber_cfos = ldpc_ber[:,0]
ldpc_ber_stos = ldpc_ber[0,:]

throughput_cfos = throughput[:,0]
throughput_stos = throughput[0,:]


# #####################################################
# Plotting BERs under CFOs
#######################################################
plt.figure(figsize=(8, 6))
plt.semilogy(cfo_arr, uncoded_ber_cfos, marker='o', linestyle='-', label='Uncoded BER')
plt.semilogy(cfo_arr, ldpc_ber_cfos, marker='s', linestyle='--', label='Coded BER')

# Labels and title
plt.xlabel('CFOs (Hz)')
plt.ylabel('BER')
plt.title('BER vs. CFO')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Show plot
plt.show()
plt.savefig('ber_cfos')

# #####################################################
# Plotting Throughput under CFOs
#######################################################
plt.figure(figsize=(8, 6))
plt.plot(cfo_arr, throughput_cfos, marker='o', linestyle='-', label='Throughput (Coded)')

# Labels and title
plt.xlabel('CFOs (Hz)')
plt.ylabel('Throughput')
plt.title('Throughput vs. CFO')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.ylim(0, 50)

# Show plot
plt.show()
plt.savefig('throughput_cfos')


# #####################################################
# Plotting BERs under STOs
#######################################################
plt.figure(figsize=(8, 6))
plt.semilogy(sto_arr, uncoded_ber_stos, marker='o', linestyle='-', label='Uncoded BER')
plt.semilogy(sto_arr, ldpc_ber_stos, marker='s', linestyle='--', label='Coded BER')

# Labels and title
plt.xlabel('STOs (ns)')
plt.ylabel('BER')
plt.title('BER vs. STO')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Show plot
plt.show()
plt.savefig('ber_stos')

# #####################################################
# Plotting Throughput under STOs
#######################################################
plt.figure(figsize=(8, 6))
plt.plot(sto_arr, throughput_stos, marker='o', linestyle='-', label='Throughput (Coded)')

# Labels and title
plt.xlabel('STOs (Hz)')
plt.ylabel('Throughput')
plt.title('Throughput vs. STO')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.ylim(0, 50)

# Show plot
plt.show()
plt.savefig('throughput_stos')

