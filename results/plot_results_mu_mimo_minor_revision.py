import sys
import os
sys.path.append(os.path.join('..'))

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Liberation Serif']
import numpy as np
import math



#############################################
# Settings
#############################################

rx_ues_arr = [1, 2, 4, 6]
scheduling = False
prediction = True
if scheduling:
    rx_ues_arr = [10]

# Define positions for the box plots
positions_all_scenarios = []
positions_scenario_1 = np.arange(1,np.size(rx_ues_arr)+1)
positions_all_scenarios.append(positions_scenario_1)
positions_all_scenarios.append(positions_scenario_1 + 5) # Shifted right for the second scenario
positions_all_scenarios.append(positions_scenario_1 + 10) # Shifted further right for the third scenario

tx_speeds = ["0.0", "20.0", "40.0", "80.0"]
rx_speeds = ["0.0", "20.0", "40.0", "80.0"]
colors = ['blue', 'red', 'green', "purple"]
start_seed = "3007"

# drops = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
drops = np.arange(1, 6)

#############################################
# KPI Handling
#############################################

ber = []
ldpc_ber = []
throughput = []
bitrate = []

for speed_idx in range(np.size(tx_speeds)):

    curr_speed = tx_speeds[speed_idx]

    temp_ber = []
    temp_ldpc_ber = []
    temp_throughput = []
    temp_bitrate = []

    for ue_arr_idx in range(np.size(rx_ues_arr)):

        ber_all_drops = []
        ldpc_ber_all_drops = []
        throughput_all_drops = []
        bitrate_all_drops = []

        for drop_idx in drops:
            
            folder_path = "results/channels_multiple_mu_mimo_minor_revision/channels_{}_{}_seed_{}_drop_{}/".format(tx_speeds[speed_idx], 
                                                                                                                    rx_speeds[speed_idx], 
                                                                                                                    int(start_seed)+drop_idx-1,  
                                                                                                                    drop_idx)
            
            if scheduling:
                file_path = os.path.join(folder_path, "mu_mimo_results_scheduling.npz")
            else:
                file_path = os.path.join(folder_path, "mu_mimo_results_UE_{}.npz".format(rx_ues_arr[ue_arr_idx]))
            if prediction and speed_idx > 0:
                file_path = os.path.join(folder_path, "mu_mimo_results_UE_{}_prediction.npz".format(rx_ues_arr[ue_arr_idx]))
            
            try:
                data = np.load(file_path)

                ber_all_drops.append(data['ber'])
                ldpc_ber_all_drops.append(data['ldpc_ber'])
                throughput_all_drops.append(data['throughput'])
                bitrate_all_drops.append(data['bitrate'])
            except:
                continue
        

            # print("Using drop: ", drop_idx)
            
            # sinr_dB_all_drops.append(data['sinr_dB'])

            # if any(math.isinf(element) for sublist in data['sinr_dB'] for element in sublist):
            #     hold = 1
        # min_length = min(len(sublist) for sublist in sinr_dB_all_drops)
        # homogeneous_sinr_dB_all_drops = [sublist[:min_length] for sublist in sinr_dB_all_drops]

        temp_ber.append(np.concatenate(ber_all_drops))
        temp_ldpc_ber.append(np.concatenate(ldpc_ber_all_drops))
        temp_throughput.append(np.concatenate(throughput_all_drops))
        temp_bitrate.append(np.concatenate(bitrate_all_drops))

        # temp_sinr_dB.append(np.mean(homogeneous_sinr_dB_all_drops, axis=0))

    # sinr_dB.append(temp_sinr_dB)
    ber.append(temp_ber)
    ldpc_ber.append(temp_ldpc_ber)
    throughput.append(temp_throughput)
    bitrate.append(temp_bitrate)

#############################################
# Plots
#############################################

throughput = np.mean(np.asarray(throughput), axis=-1)
ber = np.mean(np.asarray(ber), axis=-1)
ldpc_ber = np.mean(np.asarray(ldpc_ber), axis=-1)
bitrate = np.mean(np.asarray(bitrate), axis=-1)

print("\nThroughput: ", throughput, "\n")



############### Throughput (styled like script #2) + spectral efficiency axis

# Make sure output folder exists
os.makedirs("results/plots", exist_ok=True)

# throughput is already shape (n_series, n_groups)
n_series, n_groups = throughput.shape

# X locations for groups (centered)
x = np.arange(n_groups)

# Bar geometry: centered groups with ~0.85 total width (like script #2)
total_width = 0.85
width = total_width / n_series if n_series > 0 else total_width
offsets = (np.arange(n_series) - (n_series - 1) / 2) * width

# Labels/colors (use Matplotlib default C0, C1, ...)
labels = [f"{int(float(s))} km/h" for s in tx_speeds]
colors = [f"C{i}" for i in range(n_series)]

# Turn ON spectral-efficiency axis (Mbps / MHz = bits/s/Hz)
add_spectral_efficiency_axis = True
bandwidth_mhz = 7.68  # <-- set this to your actual bandwidth

fig, ax = plt.subplots(figsize=(12, 8))

bars_by_series = []
for k in range(n_series):
    bars = ax.bar(x + offsets[k], throughput[k], width, label=labels[k], color=colors[k])
    bars_by_series.append(bars)

# Axes styling to match script #2
ax.set_xticks(x)
ax.set_xticklabels([str(r) for r in rx_ues_arr])
ax.set_xlabel('Number of Receiver UEs')
ax.set_ylabel('Throughput (Mbps)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Right-hand spectral efficiency axis
if add_spectral_efficiency_axis:
    try:
        bw = float(bandwidth_mhz)
        if bw <= 0:
            raise ValueError("bandwidth_mhz must be > 0")
        ax2 = ax.twinx()
        # Convert y-ticks from Mbps -> bits/s/Hz using bw (Mbps / MHz = bits/s/Hz)
        y_ticks = ax.get_yticks()
        se_ticks = y_ticks / bw
        ax2.set_ylabel('Spectral Efficiency (bits/s/Hz)')
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels([f'{v:.2f}' for v in se_ticks])
        ax2.grid(False)
    except Exception as e:
        print(f'Failed to add spectral efficiency axis: {e}')

plt.tight_layout()

# Save like script #2 (PNG + EPS, tight bbox)
out_base = "results/plots/throughput_vs_rx_mean_" + "_".join([str(int(float(s))) for s in tx_speeds])
out_png = out_base + ".png"
out_eps = out_base + ".eps"
fig.savefig(out_png, dpi=200, bbox_inches='tight')
fig.savefig(out_eps, bbox_inches='tight', format='eps')

print(f"Saved figure to {out_png} and {out_eps}")

hold = 1
