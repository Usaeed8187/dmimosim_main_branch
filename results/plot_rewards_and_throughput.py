import numpy as np
import matplotlib.pyplot as plt


base_dir = 'results/channels_multiple_mu_mimo/'
 
mobility = 'high_mobility'

method = 'deqn'

rx_UEs = 4
tx_UEs = 2

drops = np.arange(1,90)

rewards = []
rl_throughput = []
sl_throughput = []
for drop_idx in drops:
    rl_rewards_data = np.load('{}channels_{}_{}/{}_rewards_drop_{}_rx_UE_{}_tx_UE_{}_imitation_none_steps_0.npz'.format(base_dir, mobility, drop_idx,method,drop_idx,rx_UEs,tx_UEs))
    rewards.append(rl_rewards_data["rewards"])

    rl_tput_data = np.load('{}channels_{}_{}/mu_mimo_results_link_adapt_rx_UE_{}_tx_UE_{}_prediction_deqn_plus_two_mode_pmi_quantization_True_imitation_none_steps_0.npz'.format(base_dir, mobility, drop_idx, rx_UEs,tx_UEs))
    rl_throughput.append(rl_tput_data['per_step_throughput'][1:])

    sl_tput_data = np.load('{}channels_{}_{}/mu_mimo_results_link_adapt_rx_UE_{}_tx_UE_{}_prediction_two_mode_pmi_quantization_True.npz'.format(base_dir, mobility, drop_idx, rx_UEs,tx_UEs))
    sl_throughput.append(sl_tput_data['per_step_throughput'][1:])

    hold = 1

rewards = np.concatenate(rewards)
rl_throughput = np.concatenate(rl_throughput)
sl_throughput = np.concatenate(sl_throughput)

window_len = 100
kernel = np.ones(window_len, dtype=float) / float(window_len)

rewards = np.convolve(np.asarray(rewards, dtype=float), kernel, mode="valid")
rl_throughput = np.convolve(np.asarray(rl_throughput, dtype=float), kernel, mode="valid")
sl_throughput = np.convolve(np.asarray(sl_throughput, dtype=float), kernel, mode="valid")

plt.figure()
plt.plot(rewards)
plt.savefig('aa')

plt.figure()
plt.plot(rl_throughput, label="DEQN + WESN")
plt.plot(sl_throughput, label="WESN")
plt.legend()
plt.savefig('bb')