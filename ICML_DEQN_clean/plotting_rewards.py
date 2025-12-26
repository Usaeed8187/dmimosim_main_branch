import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    theory_on = False
    compare_radii = False
    see_performance = False
    see_performance_avg = True
    n_pu = 12
    n_su = 3
    n_channel = 3
    n_layers = 1
    random_seed = 17
    seed_list = [10,11,12,13,14,15,16]
    radius_list = [0.7]
    pass
    
    if compare_radii:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'DEQNv3' #% n_layers
        plt.figure(figsize=(13,5))
        for my_radius in [0.1,0.3,0.5,0.9,0.99,0.7]: #[0.9,0.99]:
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
            # print(reward_SU.shape)
            window = 160
            reward_SU = np.convolve(reward_SU[0,0:10000],np.ones(window),'valid')/window
            plt.plot(reward_SU,label = 'Radius %.2f'%my_radius)
        plt.legend(loc='best')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per Episode')
        plt.grid()
        plt.show()

    if theory_on:
        file_folder = './result_theory/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        radii_list = np.load(file_folder + 'radii_list' +  '_seed%d'% random_seed+'.npy')
        error_list = np.load(file_folder + 'error_list' +  '_seed%d' % random_seed+'.npy')
        plt.figure(figsize=(8,5))
        # plt.plot(radii_list[:-1],error_list[:-1],'-o',color='orange')
        plt.plot(radii_list[:],error_list[:],'-o',color='orange')
        plt.xlabel('Spectral radius')
        plt.ylabel('One Step Approximation Error Bound')
        plt.grid()
        plt.show()
        pass
    
    if see_performance:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'DEQNv2.01' #% n_layers
        # model_name = 'ADRQN'
        plt.figure(figsize=(13,5))
        for my_radius in [0.7]: #[0.9,0.99]:
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
            # print(reward_SU.shape)
            window = 160
            for i_su in range(n_su):
                reward_SU_windowed = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
                plt.plot(reward_SU_windowed,label = 'SU %d -- Radius %.2f'%(i_su,my_radius))
        plt.legend(loc='best')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per Episode')
        plt.grid()
        plt.show()

    if see_performance_avg:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'DEQNv2.01' #% n_layers
        # model_name = 'ADRQN'
        plt.figure(figsize=(13,5))
        window = 160
        reward_SU_windowed = np.zeros((len(seed_list),len(radius_list),n_su,15000-window+1))
        for i_seed,seed in enumerate(seed_list):
            for i_radius,my_radius in enumerate(radius_list): #[0.9,0.99]:
                reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed + '_radius%.2f'%my_radius+'.npy')
                elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'_radius%.2f'%my_radius+'.npy')
                # print(reward_SU.shape)
                
                for i_su in range(n_su):
                    reward_SU_windowed[i_seed,i_radius,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        for i_su in range(n_su):
            plt.plot(reward_SU_windowed[:,0,i_su,:].mean(axis=0),label = 'SU %d -- Radius %.2f'%(i_su,my_radius))
        plt.legend(loc='best')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per Episode')
        plt.grid()
        plt.show()