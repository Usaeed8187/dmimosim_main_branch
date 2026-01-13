import numpy as np
import matplotlib.pyplot as plt


base_dir = 'results/channels_multiple_mu_mimo/'
 
mobility = 'high_mobility'

method = 'deqn'

rx_UEs = 4
tx_UEs = 2

drops = np.arange(1,32)

rewards = []
for drop_idx in drops:
    data = np.load('{}channels_{}_{}/{}_rewards_drop_{}_rx_UE_{}_tx_UE_{}_imitation_none_steps_0.npz'.format(base_dir, mobility, drop_idx,method,drop_idx,rx_UEs,tx_UEs))
    rewards.append(np.mean(data["rewards"]))

plt.figure()
plt.plot(drops, rewards)
plt.savefig('a')