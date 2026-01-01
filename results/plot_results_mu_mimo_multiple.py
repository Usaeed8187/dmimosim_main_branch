import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
import numpy as np
import math



#############################################
# Settings
#############################################

rx_ues_arr = [1, 2, 4]
scheduling = True
prediction = False
if scheduling:
    rx_ues_arr = [10]

# Define positions for the box plots
positions_all_scenarios = []
positions_scenario_1 = np.arange(1,np.size(rx_ues_arr)+1)
positions_all_scenarios.append(positions_scenario_1)
positions_all_scenarios.append(positions_scenario_1 + 5) # Shifted right for the second scenario
positions_all_scenarios.append(positions_scenario_1 + 10) # Shifted further right for the third scenario

mobilities = ['low_mobility', 'medium_mobility', 'high_mobility']
# mobilities = ['low_mobility', 'medium_mobility']

num_drops = 7

#############################################
# KPI Handling
#############################################

prediction_results = False
sinr_dB = []
ber = []
ldpc_ber = []
throughput = []

for mobility_idx in range(np.size(mobilities)):

    curr_mobility = mobilities[mobility_idx]

    temp_sinr_dB = []
    temp_ber = []
    temp_ldpc_ber = []
    temp_throughput = []

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        sinr_dB_all_drops = []
        ber_all_drops = []
        ldpc_ber_all_drops = []
        throughput_all_drops = []

        for drop_idx in np.arange(1, num_drops+1):
            
            folder_path = "results/channels_multiple_mu_mimo/channels_{}_{}/".format(curr_mobility, drop_idx)
            
            if scheduling:
                file_path = os.path.join(folder_path, "mu_mimo_results_scheduling.npz")
            else:
                file_path = os.path.join(folder_path, "mu_mimo_results_UE_{}.npz".format(rx_ues_arr[ue_arr_idx]))
            if prediction_results:
                file_path = os.path.join(folder_path, "mu_mimo_results_UE_{}_prediction.npz".format(rx_ues_arr[ue_arr_idx]))
            
            try:
                data = np.load(file_path)

                ber_all_drops.append(data['ber'])
                ldpc_ber_all_drops.append(data['ldpc_ber'])
                throughput_all_drops.append(data['throughput'])
            except:
                continue

            print("Using drop: ", drop_idx)
            
            # sinr_dB_all_drops.append(data['sinr_dB'])

            # if any(math.isinf(element) for sublist in data['sinr_dB'] for element in sublist):
            #     hold = 1
        # min_length = min(len(sublist) for sublist in sinr_dB_all_drops)
        # homogeneous_sinr_dB_all_drops = [sublist[:min_length] for sublist in sinr_dB_all_drops]

        temp_ber.append(np.concatenate(ber_all_drops))
        temp_ldpc_ber.append(np.concatenate(ldpc_ber_all_drops))
        temp_throughput.append(np.concatenate(throughput_all_drops))
        # temp_sinr_dB.append(np.mean(homogeneous_sinr_dB_all_drops, axis=0))

    # sinr_dB.append(temp_sinr_dB)
    ber.append(temp_ber)
    ldpc_ber.append(temp_ldpc_ber)
    throughput.append(temp_throughput)

#############################################
# Plots
#############################################

############################### SINR Distributions (rx nodes in phase 2) ######################################

# Method 1 for SINR Distributions (rx nodes in phase 2) (number of UEs on x-axis)
plt.figure()
colors = ['red', 'green', 'purple']

reshaped_sinr_dB = []
# Loop through the outermost dimension (mobility index)
for mobility_idx in range(len(mobilities)):
    inhomogeneous_part = []
    # Loop through the UEs dimension
    for ue_idx in range(len(rx_ues_arr)):
        # Extend inhomogeneous_part by the ue_idx-th list (inhomogeneous data)
        inhomogeneous_part.append(np.concatenate(sinr_dB[mobility_idx][ue_idx]))
    
    # Append the reshaped inhomogeneous part for the current mobility index
    reshaped_sinr_dB.append(inhomogeneous_part)


    
for mobility_idx in range(np.size(mobilities)):
    plt.boxplot(reshaped_sinr_dB[mobility_idx], positions=positions_all_scenarios[mobility_idx], widths=0.6, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colors[mobility_idx], color=colors[mobility_idx]),
                medianprops=dict(color='black'))
plt.xticks(np.concatenate(positions_all_scenarios), rx_ues_arr * 3)
legend_elements = [plt.Line2D([0], [0], color='red', lw=4, label='Scenario 1'),
                   plt.Line2D([0], [0], color='green', lw=4, label='Scenario 2'),
                   plt.Line2D([0], [0], color='purple', lw=4, label='Scenario 3')]
plt.legend(handles=legend_elements, title='Scenarios', loc='upper right')
plt.grid(True)
plt.xlabel('Number of UEs')
plt.ylabel('SINR')
plt.title('SINR (dB)')
# plt.ylim(5.5, 5.6)
plt.savefig("results/plots/SINR_MU_MIMO_updated")


# # Method 2 for SINR Distributions (rx nodes in phase 2) (mobility on x-axis and only showing best selection)
# num_categories = len(mobilities)
# mean_snr_db_arr = np.zeros(num_categories)
# UE_ind = 2
# for i in range(num_categories):
#     mean_snr_db_arr[i] = np.mean(sinr_dB[i][UE_ind])
# x = np.arange(num_categories)
# bar_width = 0.35
# x_labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']
# plt.figure()
# plt.bar(x, np.asarray(mean_snr_db_arr), width=bar_width, label='MU MIMO', color='#4F81BD')
# plt.xticks(x, x_labels)
# plt.grid(True)
# plt.ylabel('SINR')
# plt.title('SINR (dB)')
# plt.savefig("results/plots/SINR_MU_MIMO")


hold = 1