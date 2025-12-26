from DSA_ENV import DSA_env
from RayTraModel import *
from DQN_RC_new_WESN import DeepWESNQNetwork
import numpy as np
import copy
import dill
import time
from itertools import product
from Theory_incorporation import generate_DQNs, BiasError, VarianceError



if __name__ == "__main__":

    for random_seed in range(10,10):
            # Note: 3 lines of plt.show() have been commented out, two in RayTraModel.py and one in DSA_ENV.py
        # Use random_seed=14 for 1 SU and 12 PUs
        # random_seed = 10
        np.random.seed(random_seed)

        # Number of channel, Number of PUs per channel, Number of SUs
        n_channel = 3
        n_pu = 12
        n_su = 3

        num_neurons = 16
        input_window_length = 4
        output_window_length = 4

        dim_actions = n_channel * 2  # The action space size
        dim_states = 1 + n_channel   # The state space size (sensed energy + sensor)
        learning_method = 'double'

        theory_on = False

        my_radius = 0.9



        '''
        Read ray-tracing data
        '''
        site = 'Mumbai'
        rayTraEnv = RayTraModel(site)

        # Select the simulation area and cells
        center = np.array([-90, 180])
        area = np.array([400, 400])
        selected_cell = np.array([22, 23, 24, 25, 35, 36, 37, 38])
        bs_cell = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        channel_bs = np.array([[0], [1]])
        PUT_BS = np.array([1])
        SUT_BS = np.array([0])

        if n_channel == 2:
            selected_cell = np.array([22, 24, 35, 38])
            bs_cell = np.array([[0], [1], [2], [3]])
            channel_bs = np.array([[0, 3], [1, 2]])
            PUT_BS = np.array([3, 2])
            SUT_BS = np.array([0, 1])

        if n_channel == 3:
            selected_cell = np.array([22, 23, 24, 35, 36, 38])
            bs_cell = np.array([[0], [1], [2], [3], [4], [5]])
            channel_bs = np.array([[0, 3], [1, 4], [2, 5]])
            PUT_BS = np.array([3, 4, 5])
            SUT_BS = np.array([0, 1, 2])


        n_bs = bs_cell.shape[0]
        n_cell = selected_cell.size

        rayTraEnv.show_range(area, center) # Before reducing data
        rayTraEnv.reduceData(center, area, selected_cell) # Reduce data
        #rayTraEnv.show_range(area, np.array([0, 0]))  # After reducing data

        '''
        for tilt_list in product(np.arange(13), repeat=n_bs):
            tilt = np.asarray(tilt_list)
            print('tilt: ', tilt)
            channel.SetTilt(tilt, cell_bs)  # Set the tilt angle
            channel.show_BSRange(cell_bs)
            channel.show_SINR()
        '''

        tilt = np.array([2, 0])
        tilt = np.array([2, 0, 0, 0])
        tilt = np.array([2, 2, 2, 0, 0, 0])
        rayTraEnv.SetTilt(tilt, bs_cell)  # Set the tilt angle
        rayTraEnv.show_BSRange_SINR(channel_bs)
        #rayTraEnv.show_channel_gain(channel_bs)


        # training parameters
        batch_size = 500 # 300
        total_episode = batch_size * 300 # 200
        epsilon_update_period = batch_size * 1
        e_greedy_start = 0.7  # 0.5
        if theory_on: e_greedy_start = 0
        e_greedy_end = 0.9 # 0.9
        e_increase = (e_greedy_end - e_greedy_start) / 290 # 190
        learning_rate = 0.01 #0.01 #
        epsilon = np.ones(n_su) * e_greedy_start


        # Initialize some record values
        reward_SU = np.zeros((n_su, total_episode))

        # Bin_SU_2 = np.zeros((n_su, total_episode))
        # Brec_SU_2 = np.zeros((n_su, total_episode))
        # Bout_SU_2 = np.zeros((n_su, total_episode))
        # Bin_SU_F = np.zeros((n_su, total_episode))
        # Brec_SU_F = np.zeros((n_su, total_episode))
        # Bout_SU_F = np.zeros((n_su, total_episode))

        access_channel = np.zeros((n_channel, total_episode))
        fail_channel = np.zeros((n_channel, total_episode))
        success_channel = np.zeros((n_channel, total_episode))

        access_SU = np.zeros((n_su, total_episode))
        success_SU = np.zeros((n_su, total_episode))
        fail_SU = np.zeros((n_su, total_episode))

        bps_SU = np.zeros((n_su, total_episode))
        bps_channel = np.zeros((n_channel, total_episode))


        # Initialize the environment
        env = DSA_env(n_channel, n_pu, n_su, channel_bs, PUT_BS, SUT_BS, rayTraEnv)

        # Store the initial environment
        file = open('env_PU%d_SU%d_seed%d.obj' % (n_pu, n_su, random_seed), 'wb')
        dill.dump(env, file)
        file.close()

        # random_seed = 6 # commented out by Ramin
        # np.random.seed(random_seed) # Commented out by Ramin


        # Initialize the sensor
        active_sensor = np.zeros((n_su, n_channel)).astype(np.int32)
        initial_sensed_channel = np.random.choice(n_channel, n_su)
        for k in range(n_su):
            active_sensor[k, initial_sensed_channel[k]] = 1


        # Initialize the DQN_RC
        DEQN_list = []
        n_layers = 1
        if theory_on: list_of_su_DQNs = []
        for k in range(n_su):
            DEQN_tmp = DeepWESNQNetwork(dim_actions, dim_states,
                                        input_window_length, output_window_length,
                                        batch_size, n_layers,
                                        nInternalUnits=num_neurons,
                                        reward_decay=0.9,
                                        e_greedy=e_greedy_start,
                                        lr=learning_rate,
                                        random_seed=random_seed,
                                        spectral_radius=my_radius)
            DEQN_list.append(DEQN_tmp)
            if theory_on:
                list_of_radii = [0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7,0.9,0.99]
                list_of_radii = [i/10 for i in range(1,10,1)]
                list_of_DQNs = generate_DQNs(DEQN_list[k],list_of_radii,random_seed)
                list_of_su_DQNs.append(list_of_DQNs)


        env.render(0)
        # SUs sense the environment and get the sensing result (contains sensing errors)
        observation = env.sense(active_sensor, 0)

        t = time.time()
        training_time = 0
        
        pretrain_episodes = batch_size
        # for step in range(pretrain_episodes):
        #     
        #     pass
        
        for step in range(total_episode):

            # SU choose action based on observation
            action = np.zeros(n_su).astype(np.int32)
            for k in range(n_su):
                action[k] = DEQN_list[k].choose_action(observation[k, :])
                if theory_on:
                    for DQN in list_of_su_DQNs[k]:
                        DQN.choose_action(observation[k, :])

            # SU take action and get the reward
            reward = env.access(action, step)

            # Record values
            reward_SU[:, step] = reward

            fail_channel[:, step] = env.fail_channel
            success_channel[:, step] = env.success_channel

            access_SU[:, step] = env.access_SU
            success_SU[:, step] = env.success_SU
            fail_SU[:, step] = env.fail_SU

            # update the SU sensors
            active_sensor = env.render_sensor(action)

            # SU sense the environment and get the sensing result (contains sensing errors)
            if (step+1) < total_episode:
                env.render(step + 1)
                observation_ = env.sense(active_sensor, step+1)

            # Activate the SU's target_net's state based on observation_
            for k in range(n_su):
                DEQN_list[k].activate_target_net(observation_[k, :])
                if theory_on:
                    for DQN in list_of_su_DQNs[k]:
                        DQN.activate_target_net(observation_[k, :])

            # Store one episode (s, a, r, s')
            for k in range(n_su):
                state = observation[k, :]
                state_ = observation_[k, :]
                DEQN_list[k].store_transition(state, action[k], reward[k], state_)
                if theory_on:
                    for DQN in list_of_su_DQNs[k]:
                        DQN.store_transition(state, action[k], reward[k], state_)
            
            # Each SU learns their DQN model
            if ((step + 1) % (batch_size) == 0):
                for k in range(n_su):
                    
                    ####
                    if (step+1 == batch_size) and theory_on:
                        list_of_DQNs = list_of_su_DQNs[k]
                        total_error = np.zeros(len(list_of_DQNs))
                        for i,DQN in enumerate(list_of_DQNs):
                            total_error[i] = 4 * BiasError(DQN) + VarianceError(DQN,1/(1-DQN.gamma),2.5)
                            print(list_of_radii[i],'=>',total_error[i])
                        file_folder = './result_theory/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
                        import os
                        os.makedirs(file_folder,exist_ok=True)
                        np.save(file_folder + 'radii_list' +  '_seed%d'% random_seed + 'numNeu%d'%num_neurons, list_of_radii)
                        np.save(file_folder + 'error_list' +  '_seed%d' % random_seed+ 'numNeu%d'%num_neurons, total_error)

                        exit()
                        pass
                        #

                    ####
                    current_time = time.time()
                    DEQN_list[k].learn_new(batch_size, step, method=learning_method)
                    training_time += (time.time() - current_time)
                    DEQN_list[k].epsilon = epsilon[k]
                    '''
                    if (epsilon[k] >= 0.8):
                        DEQN_list[k].update_lr(0.001)
                    else:
                        DEQN_list[k].update_lr(0.01)
                    '''
                    if np.mean(reward_SU[:, step+1-batch_size:step+1]) > 0.6:
                        if DEQN_list[k].eval_net.lr != 0.001:
                            DEQN_list[k].update_lr(0.001)
                            print('================ updated learning rate to 0.001 ================')
                # Check the norms and use the covering number upper bound


            # for k in range(n_su):
            #     W_rec = DEQN_list[k].sess.run(DEQN_list[k].rnnCell1_eval.trainable_variables[0][-DEQN_list[k].nInternalUnits:,:])
            #     Brec_SU_2[k,step] = np.linalg.norm(W_rec,ord=2)
            #     Brec_SU_F[k,step] = np.linalg.norm(W_rec,ord='fro')
            #     W_in = DEQN_list[k].sess.run(DEQN_list[k].rnnCell1_eval.trainable_variables[0][:-DEQN_list[k].nInternalUnits,:])
            #     Bin_SU_2[k,step] = np.linalg.norm(W_in,ord=2)
            #     Bin_SU_F[k,step] = np.linalg.norm(W_in,ord='fro')
            #     W_out = DEQN_list[k].sess.run(DEQN_list[k].eval_output_weights)
            #     Bout_SU_2[k,step] = np.linalg.norm(W_out,ord=2)
            #     Bout_SU_F[k,step] = np.linalg.norm(W_out,ord='fro')

            if ((step+1) % batch_size == 0):
                index = np.arange(step+1-batch_size, step+1)
                print('Training episode = %d' % int((step + 1)/batch_size))
                print('Training period = %d' % (step + 1))
                print('SU: success = %d;  fail = %d;  access = %d' %
                    (np.sum(success_SU[:, index]), np.sum(fail_SU[:, index]), np.sum(access_SU[:, index])))
                for c in range(n_channel):
                    print('Channel%d: fail = %d;  success = %d' %
                        (c+1, np.sum(fail_channel[c, index]), np.sum(success_channel[c, index])))
                print('total_reward = %.4f' % (np.sum(reward_SU[:, index])/batch_size))
                rewardPrint = ''
                for k in range(n_su):
                    rewardPrint = rewardPrint + 'SU %d '%(k+1) + \
                                '%.2f, '% (np.sum(reward_SU[k, index])/batch_size)
                print(rewardPrint)
                print('random_seed=',random_seed)

            # Update epsilon
            if ((step + 1) % epsilon_update_period == 0):
                for k in range(n_su):
                    epsilon[k] = min(e_greedy_end, epsilon[k] + e_increase)
                    #epsilon[k] = e_greedy_end


            # swap observation
            observation = observation_

        elapsed = time.time() - t
        print('Elapsed Time = %.4f sec' % elapsed)
        print('Total Training Time = %.4f sec' % training_time)
        observation_last = copy.deepcopy(observation)
        env_last = copy.deepcopy(env)

        '''
        for k in range(n_su):
            file = open('DQN_RC_SU%d.obj'%k, 'wb')
            dill.dump(DQN_RC_list[k], file)
            file.close()
        '''

        file_folder = './result_WESN/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
        # model_name = 'DEQN%d' % n_layers # lr = 0.001
        # model_name = 'DEQNv2' # lr = 0.01
        model_name = 'DEQNv2.01_%d'%num_neurons # lr = 0.01 then 0.001
        # model_name = 'DEQNv3' # lr = 0.01 and norm(W_in) was fixed to have the same raduis as W_rec (or W for that matter)
        print(model_name)

        import os
        os.makedirs(file_folder,exist_ok=True)
        # WeightNormsFolder = file_folder+'/norms/'
        # os.makedirs(WeightNormsFolder,exist_ok=True)
        np.save(file_folder + 'reward_SU_' + model_name + '_seed%d'% random_seed + '_radius%.2f'%my_radius, reward_SU)
        np.save(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed+'_radius%.2f'%my_radius, elapsed)

        # for name, weight in [('in_2',Bin_SU_2),('rec_2',Brec_SU_2),('out_2',Bout_SU_2),('in_F',Bin_SU_F),('rec_F',Brec_SU_F),('out_F',Bout_SU_F)]:
        #     np.save(WeightNormsFolder + 'norm_'+ name + '_' + model_name + '_seed%d' % random_seed, weight)
