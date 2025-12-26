import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt


if __name__=='__main__':
    theory_on = False
    compare_radii = False
    see_performance = False
    see_performance_avg = False
    see_performance_avg_by_su = False
    see_success_fail_prob = True
    save_figures = False
    n_pu = 12
    n_su = 3
    n_channel = 3
    n_layers = 1
    num_neurons = 16
    random_seed = 1
    radius_list = [0.7]
    if n_su == 3:
        seed_list= [10,11,12,13,14,15,16,17]
    elif n_su == 1:
        seed_list= [1,2,3,4,5,7,8,9,10]
    shape_list = ['-','--','-.']
    episode_size = 500
    pass

    if save_figures:
        plt.rcParams.update({'font.size': 14})
        plt.rcParams["font.family"] = "Times New Roman"
    
    if compare_radii:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'DEQNv3' #% n_layers
        model_name = 'DEQNv2.01_%d'%num_neurons
        plt.figure(figsize=(10,5))
        for my_radius in [0.3,0.5,0.7,0.9]: #[0.9,0.99]:
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
            # print(reward_SU.shape)
            window = 360
            reward_SU = np.convolve(reward_SU[0,0:10000],np.ones(window),'valid')/window
            # reward_SU_by_episode = np.reshape(reward_SU,[-1,episode_size])
            # reward_SU_avg = reward_SU_by_episode.mean(axis=1)
            # plt.plot(reward_SU_avg[:40],label = 'Radius %.2f'%my_radius)
            plt.plot(reward_SU,label = 'Radius %.2f'%my_radius)
        plt.legend(loc='best')
        plt.xlabel('Decision Step')
        plt.ylabel('Rolling Average Reward')
        plt.grid()
        if save_figures: plt.savefig('../Figures/PU4SU1rewardRolling_16.eps',format='eps')
        plt.show()

    if theory_on:
        file_folder = './result_theory/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        radii_list = np.load(file_folder + 'radii_list' +  '_seed%d'% random_seed+ 'numNeu%d'%num_neurons+'.npy')
        error_list = np.load(file_folder + 'error_list' +  '_seed%d' % random_seed+ 'numNeu%d'%num_neurons+'.npy')
        plt.figure(figsize=(8,5))
        # plt.plot(radii_list[:-1],error_list[:-1],'-o',color='orange')
        plt.plot(radii_list[:],error_list[:],'-o',color='orange')
        plt.xlabel('Spectral radius')
        plt.ylabel('One Step Approximation Error Bound')
        plt.grid()
        plt.savefig('../Figures/BiasPlusVariance_16_theory.eps',format='eps')
        plt.show()
        pass
    
    if see_performance:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        # model_name = 'DEQN1' #% n_layers
        model_name = 'DEQNv2'

        # model_name = 'DRQN1_numNue%d'%32
        plt.figure(figsize=(13,5))
        window = 160
        


        # for my_radius in [0.7]: #[0.9,0.99]:
        #     reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
        #     elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
        #     # print(reward_SU.shape)
        ##     window = 160
        #     for i_su in range(n_su):
        #         reward_SU_windowed = np.convolve(reward_SU[i_su,0:3500],np.ones(window),'valid')/window
        #         plt.plot(reward_SU_windowed,label = 'SU %d -- Radius %.2f'%(i_su,my_radius)+' lr=0.01')
        
        # model_name = 'DEQNv2.1'
        # for my_radius in [0.7]: #[0.9,0.99]:
        #     reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
        #     elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
        #     # print(reward_SU.shape)
        ##     window = 160
        #     for i_su in range(n_su):
        #         reward_SU_windowed = np.convolve(reward_SU[i_su,0:3500],np.ones(window),'valid')/window
        #         plt.plot(reward_SU_windowed,label = 'SU %d -- Radius %.2f'%(i_su,my_radius)+' lr=0.1 first, then 0.01')

        model_name = 'DEQNv2.01'
        for my_radius in [0.7]: #[0.9,0.99]:
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius+'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius+'.npy')
            # print(reward_SU.shape)
            # window = 160
            reward_SU_windowed=0
            for i_su in range(n_su):
                reward_SU_windowed += np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
                # plt.plot(reward_SU_windowed,label = 'SU %d -- Radius %.2f'%(i_su,my_radius)+' lr=0.1 first, then 0.01')
            plt.plot(reward_SU_windowed/n_su,label = 'SU - mean -- Radius %.2f'%(my_radius)+' lr= 0.01')


        random_seed=15
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'ADRQN' #% n_layers
        reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed +'.npy')
        elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'.npy')
        reward_SU_windowed=0
        for i_su in range(n_su):
            
            reward_SU_windowed += np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
            # plt.plot(reward_SU_windowed,'--',label = 'SU %d -- ADRQN'%(i_su))
        plt.plot(reward_SU_windowed/n_su,'--',label = 'SU-mean -- ADRQN')
        
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
        
        
        # plt.plot(reward_SU_windowed[:,0,:,:].mean(axis=1).mean(axis=0),label = 'mean over SUs and seeds -- DEQN')#Radius %.2f'%(my_radius))
        plt.plot(reward_SU_windowed[:,0,:,:].mean(axis=1).mean(axis=0),label = 'DEQN')
        
        reward_SU_windowed = np.zeros((len(seed_list),n_su,15000-window+1))
        model_name = 'ADRQN'
        for i_seed,seed in enumerate(seed_list):
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed +'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'.npy')
                # print(reward_SU.shape)
                
            for i_su in range(n_su):
                reward_SU_windowed[i_seed,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        
        # plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0),label = 'mean over SUs and seeds -- ADRQN') #%s'%(model_name))
        plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0),label = 'ADRQN')

        reward_SU_windowed = np.zeros((len(seed_list),n_su,15000-window+1))
        model_name = 'DRQN1'
        for i_seed,seed in enumerate(seed_list):
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed +'.npy')
            elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'.npy')
                # print(reward_SU.shape)
                
            for i_su in range(n_su):
                reward_SU_windowed[i_seed,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        
        # plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0),label = 'mean over SUs and seeds -- DRQN') #%s'%(model_name))
        plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0),label = 'DRQN')
        
        plt.legend(loc='best')
        plt.title('Reward received with %d PUs and %d SUs for %d total channels to select from'%(n_pu,n_su,n_channel))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per Episode')
        plt.grid()
        plt.show()
        pass
    
    if see_performance_avg_by_su:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        model_name = 'DEQNv2.01' #% n_layers
        # model_name = 'ADRQN'
        plt.figure(figsize=(13,5))
        window = 160
        reward_SU_windowed = np.zeros((len(seed_list),len(radius_list),n_su,15000-window+1))
        elapsed=0
        for i_seed,seed in enumerate(seed_list):
            for i_radius,my_radius in enumerate(radius_list): #[0.9,0.99]:
                reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed + '_radius%.2f'%my_radius+'.npy')
                elapsed += np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'_radius%.2f'%my_radius+'.npy')
                
                # print(reward_SU.shape)
                
                for i_su in range(n_su):
                    reward_SU_windowed[i_seed,i_radius,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        print('DEQN Simulation time:',elapsed/(len(seed_list)*len(radius_list)),'seconds')
        for i_su in range(n_su):
            plt.plot(reward_SU_windowed[:,0,:,:].mean(axis=1).mean(axis=0), shape_list[i_su],
                     label = 'mean over seeds -- SU%d Radius %.2f'%(i_su+1,my_radius), color='blue')
        
        reward_SU_windowed = np.zeros((len(seed_list),n_su,15000-window+1))
        model_name = 'ADRQN'
        elapsed = 0
        for i_seed,seed in enumerate(seed_list):
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed +'.npy')
            elapsed += np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'.npy')
                # print(reward_SU.shape)
                
            for i_su in range(n_su):
                reward_SU_windowed[i_seed,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        print('ADRQN Simulation time:',elapsed/(len(seed_list)*len(radius_list)),'seconds')
        for i_su in range(n_su):
            plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0), shape_list[i_su],
                     label = 'mean over seeds -- SU%d %s'%(i_su+1,model_name),color = 'orange')

        reward_SU_windowed = np.zeros((len(seed_list),n_su,15000-window+1))
        model_name = 'DRQN1'
        elapsed = 0
        for i_seed,seed in enumerate(seed_list):
            reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed +'.npy')
            elapsed += np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'.npy')
                # print(reward_SU.shape)
                
            for i_su in range(n_su):
                reward_SU_windowed[i_seed,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
        
        print('DRQN Simulation time:',elapsed/(len(seed_list)*len(radius_list)),'seconds')
        for i_su in range(n_su):
            plt.plot(reward_SU_windowed.mean(axis=1).mean(axis=0), shape_list[i_su],
                     label = 'mean over seeds -- SU%d %s'%(i_su+1,model_name),color = 'red')

        plt.legend(loc='best')
        plt.title('Reward received with %d PUs and %d SUs for %d total channels to select from'%(n_pu,n_su,n_channel))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward per Episode')
        plt.grid()
        plt.show()


    if see_success_fail_prob:
        file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        file_folder_WESN = './result_WESN/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        # file_folder = 'ICML_DEQN_clean/result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        
        # model_name = 'ADRQN'

        # my_radius = 0.7
        # reward_SU = np.load(file_folder + 'reward_SU_' + model_name + '_seed%d'% seed + '_radius%.2f'%my_radius+'.npy')

        compare_drqn_numUnits = False
        median_filtering = False

        fig_success,ax_success = plt.subplots(figsize=(15,8))
        fig_reward , ax_reward = plt.subplots(figsize=(15,8))
        # ax_failure = plt.figure(figsize=(13,5))
        episode_size = 500
        window = 160
        # colors = ['blue','cyan','orange','red','green','brown']
        colors = ['blue','black','red','green','brown']
        # colors = ['blue','orange','red']
        if n_su==3:
            names = ['DEQN','ADRQN','DRQN','DWEQN']
        elif n_su==1:
            if compare_drqn_numUnits:
                names = ['DEQN16','DRQN64','DRQN32','DRQN16']#,'BasicDRQN']
            else:
                names = ['DEQN64','ADRQN%d'%num_neurons,'DRQN%d'%num_neurons,'Myopic-Est', 'DWEQN']#,'BasicDRQN']
            # names = ['DEQN64','ADRQN%d'%num_neurons,'DRQN%d'%num_neurons,'Myopic-Est','BasicDRQN_numNeu%d'%num_neurons]
        # names = ['DEQN%d'%num_neurons,'ADRQN%d'%num_neurons,'DRQN%d'%num_neurons,'Myopic-Est']#,'BasicDRQN']
        # names = ['DEQN64','Myopic Est']
        # for i_name,model_name in enumerate(['DEQNv2.01','ADRQN','DRQN1']): #% n_layers
        if n_su==3:
            model_name_list = ['DEQNv2.01','ADRQN','DRQN1','DEQNv2.01_%d'%num_neurons]#,'basicDRQN']
        elif n_su==1:
            if compare_drqn_numUnits:
                model_name_list = ['DEQNv2.01_16','DRQN1_numNeu64','DRQN1_numNeu32','DRQN1_numNeu16']
            else:
                model_name_list = ['DEQNv2.01_%d'%num_neurons,'ADRQN_numNeu%d'%num_neurons,'DRQN1_numNeu%d'%num_neurons, 'Myopic_Est','DEQNv2.01_%d'%num_neurons]#,'basicDRQN']
            # model_name_list = ['DEQNv2.01_%d'%num_neurons,'ADRQN_numNeu%d'%num_neurons,'DRQN1_numNeu%d'%num_neurons, 'Myopic_Est','basicDRQN_numNeu%d'%num_neurons]
        # model_name_list = ['DEQNv2.01_%d'%num_neurons,'ADRQN_numNeu%d'%num_neurons,'DRQN1_numNeu%d'%num_neurons, 'Myopic_Est']#,'basicDRQN']
        # model_name_list = ['DEQNv2.01','Myopic_Est']
        reward_SU_by_step = np.zeros([len(model_name_list),len(seed_list),len(radius_list),n_su,150000])

        for i_name,model_name in enumerate(model_name_list): #% n_layers
        # for i_name,model_name in enumerate(['DEQNv3','ADRQN','DRQN1']): #% n_layers
            # reward_SU_windowed = np.zeros((len(seed_list),len(radius_list),n_su,15000-window+1))
            # reward_SU_by_step = np.zeros([len(seed_list),len(radius_list),n_su,150000])
            for i_seed,seed in enumerate(seed_list):
                # if model_name == 'DEQNv3': seed = 1
                for i_radius,my_radius in enumerate(radius_list): #[0.9,0.99]:
                    file_folder_new = file_folder
                    if i_name==0:# or i_name==1:
                        file_name = 'reward_SU_' + model_name + '_seed%d'% seed + '_radius%.2f'%my_radius+'.npy'
                    elif i_name<len(model_name_list)-1:
                        file_name='reward_SU_' + model_name + '_seed%d'% seed +'.npy'
                    
                    else: 
                        file_folder_new = file_folder_WESN
                        my_radius = 0.9
                        file_name = 'reward_SU_' + model_name + '_seed%d'% seed + '_radius%.2f'%my_radius+'.npy'
                    reward_SU_by_step[i_name,i_seed,i_radius,:] = np.load(file_folder_new + file_name)
                    # elapsed = np.load(file_folder + 'elapsedTime_' + model_name + '_seed%d' % seed +'_radius%.2f'%my_radius+'.npy')
                    # print(reward_SU.shape)
                    
                    # for i_su in range(n_su):
                    #     reward_SU_windowed[i_seed,i_radius,i_su,:] = np.convolve(reward_SU[i_su,0:15000],np.ones(window),'valid')/window
            
            reward_SU_by_step_mean = np.mean(reward_SU_by_step,axis=1)[i_name,0,:,:] # average over seeds, take the first radius
            # reward_SU_by_episode = np.zeros([reward_SU_by_step_mean.shape[0],reward_SU_by_step_mean.shape[1]//episode_size])
            # for i in range(reward_SU_by_step_mean.shape[-1]//episode_size):
            #     reward_SU_by_episode[:,1] = np.mean(reward_SU_by_step_mean[:,i*episode_size:(i+1)*episode_size],axis=1)
            reward_shape = reward_SU_by_step[i_name].shape
            reward_SU_by_episode = np.reshape(reward_SU_by_step[i_name],reward_shape[:-1]+(reward_shape[-1]//episode_size,episode_size))
            success_prob=np.sum(reward_SU_by_episode==1,axis=-1)/episode_size
            failure_prob=np.sum(reward_SU_by_episode==-1,axis=-1)/episode_size
            idle_prob=np.sum(reward_SU_by_episode==0,axis=-1)/episode_size
            reward_SU_by_episode_mean = reward_SU_by_episode.mean(axis=-1).mean(axis=0)[0] # mean over seeds, first radius
            success_prob_mean = success_prob.mean(axis=0)[0,:,:] # mean over seeds
            failure_prob_mean = failure_prob.mean(axis=0)[0,:,:] # mean over seeds
            # for i_su in range(n_su):
            #     ax_success.plot(success_prob_mean[i_su],'-',color=colors[i_name],label=f'{names[i_name]} Success Probability, SU {i_su}')
            # ax_success.plot(failure_prob_mean[0],'--',color=colors[i_name],label=f'{names[i_name]} Faliure Probability')
            # ax_reward.plot(reward_SU_by_episode_mean[0],'-',color=colors[i_name],label=f'{names[i_name]} reward')

            ax_success.plot(failure_prob_mean.mean(axis=0)[:200],'--',color=colors[i_name],label=f'{names[i_name]} Failure Probability')
            
            if median_filtering:
                alpha_1 = 0.2
                ax_reward.plot(reward_SU_by_episode_mean.mean(axis=0)[:200],'-',color=colors[i_name],alpha=alpha_1)
                ax_reward.plot(medfilt(reward_SU_by_episode_mean.mean(axis=0),5)[:200],'-',color=colors[i_name],label=f'{names[i_name]} reward')
            else:
                ax_reward.plot(reward_SU_by_episode_mean.mean(axis=0)[:200],'-',color=colors[i_name],label=f'{names[i_name]} reward')
            ax_success.plot(success_prob_mean.mean(axis=0)[:200],'-',color=colors[i_name],label=f'{names[i_name]} Success Probability')
        
        ax_success.set_xlabel('Episode (=500 steps)',fontsize=20)
        ax_reward.set_xlabel('Episode (=500 steps)',fontsize=20)

        if save_figures: ax_success.set_xlim([-10,300])
        
        ax_success.set_ylabel('Probability',fontsize=20)
        ax_reward.set_ylabel('Reward',fontsize=20)
        
        ax_success.grid()
        ax_reward.grid()

        ax_success.legend(loc='best')
        ax_reward.legend(loc='best')

        if save_figures:
            if n_su==3:
                fig_reward.savefig('../Figures/Reward_averaged.eps',format='eps')
                fig_success.savefig('../Figures/Success_Fail_prob.eps',format='eps')
            elif n_su==1:
                fig_reward.savefig('../Figures/reward_4PU_1SU_16neurons.eps',format='eps')
                fig_success.savefig('../Figures/success_failure_4PU_1SU_16neurons.eps',format='eps')

        plot_threshold = False
        if plot_threshold:
            if n_su == 1:
                margin = 0.2
                ylow = -0.5
                yhigh = 1
            else:
                margin = 0.25
                ylow = -1.5
                yhigh = 2
            reward_shape = reward_SU_by_step.shape
            reward_SU_by_episode = np.reshape(reward_SU_by_step,reward_shape[:-1]+(reward_shape[-1]//episode_size,episode_size))
            reward_SU_by_episode_mean = np.mean(reward_SU_by_episode,axis=-1) # average over an episode
            reward_SU_by_episode_mean = np.mean(reward_SU_by_episode_mean,axis=3) # average over users
            reward_SU_by_episode_mean = np.mean(reward_SU_by_episode_mean,axis=2) # average over radius
            reward_SU_by_episode_mean = np.mean(reward_SU_by_episode_mean,axis=1) # average over seeds
            end = 200
            ep = np.arange(end)
            fig_thresh_DRQN,ax_thresh_DRQN = plt.subplots(figsize=(15,8))
            # drqn_difference = np.mean(2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[2])/(0.001+reward_SU_by_episode_mean[0]+reward_SU_by_episode_mean[2]),axis=(0,1,2))
            drqn_difference = 2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[2])/(reward_SU_by_episode_mean[0]+reward_SU_by_episode_mean[2])
            drqn_difference_medfilt = medfilt(drqn_difference,5)
            # drqn_difference = np.mean(2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[2]),axis=(0,1,2))
            ax_thresh_DRQN.plot(ep,drqn_difference[:end],
                                label = 'Ratio of Reward Difference Betweeen DEQN and DRQN',)
            ax_thresh_DRQN.plot(ep,drqn_difference_medfilt[:end],
                                color='red')
            ax_thresh_DRQN.plot(ep,margin*np.ones(end),'--',color='orange')
            ax_thresh_DRQN.plot(ep,-margin*np.ones(end),'--',color='orange')
            ax_thresh_DRQN.set_xlabel('Episode (=500 steps)',fontsize=20)
            ax_thresh_DRQN.set_ylabel('Reward Difference',fontsize=20)
            ax_thresh_DRQN.set_ylim(ylow,yhigh)
            ax_thresh_DRQN.legend()
            
            fig_thresh_ADRQN,ax_thresh_ADRQN = plt.subplots(figsize=(15,8))
            # adrqn_difference = np.mean(2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[1])/(0.001+reward_SU_by_episode_mean[0]+reward_SU_by_episode_mean[1]),axis=(0,1,2))
            adrqn_difference = 2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[1])/(reward_SU_by_episode_mean[0]+reward_SU_by_episode_mean[1])
            adrqn_difference_medfilt = medfilt(adrqn_difference,5)
            # adrqn_difference = np.mean(2*(reward_SU_by_episode_mean[0]-reward_SU_by_episode_mean[1]),axis=(0,1,2))
            ax_thresh_ADRQN.plot(ep,adrqn_difference[:end],
                                 label = 'Ratio of Reward Difference Betweeen DEQN and ADRQN')
            ax_thresh_ADRQN.plot(ep,adrqn_difference_medfilt[:end],
                                 color='red')
            ax_thresh_ADRQN.plot(ep,margin*np.ones(end),'--',color='orange')
            ax_thresh_ADRQN.plot(ep,-margin*np.ones(end),'--',color='orange')
            ax_thresh_ADRQN.set_xlabel('Episode (=500 steps)',fontsize=20)
            ax_thresh_ADRQN.set_ylabel('Reward Difference',fontsize=20)
            ax_thresh_ADRQN.set_ylim(ylow,yhigh)
            ax_thresh_ADRQN.legend()

        # plt.legend(loc='best')
        # plt.show()

        