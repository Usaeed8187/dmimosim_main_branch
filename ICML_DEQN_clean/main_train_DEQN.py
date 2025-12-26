from DSA_ENV import DSA_env
from RayTraModel import *
from DQN_RC_new import DeepQNetwork
import numpy as np
import copy
import dill
import time
from itertools import product
from typing import List



if __name__ == "__main__":

    random_seed = 1
    np.random.seed(random_seed)

    # Number of channel, Number of PUs per channel, Number of SUs
    n_channel = 3
    n_pu = 4
    n_su = 1

    dim_actions = n_channel * 2  # The action space size
    dim_states = 1 + n_channel   # The state space size (sensed energy + sensor)

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
    e_greedy_end = 0.9 # 0.9
    e_increase = (e_greedy_end - e_greedy_start) / 290 # 190
    learning_rate = 0.001 #0.01
    epsilon = np.ones(n_su) * e_greedy_start


    # Initialize some record values
    reward_SU = np.zeros((n_su, total_episode))

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

    random_seed = 6
    np.random.seed(random_seed)


    # Initialize the sensor
    active_sensor = np.zeros((n_su, n_channel)).astype(np.int32)
    initial_sensed_channel = np.random.choice(n_channel, n_su)
    for k in range(n_su):
        active_sensor[k, initial_sensed_channel[k]] = 1


    # Initialize the DQN_RC
    DEQN_list:List[DeepQNetwork] = []
    n_layers = 1
    for k in range(n_su):
        DEQN_tmp = DeepQNetwork(dim_actions, dim_states, batch_size, n_layers,
                                reward_decay=0.9,
                                e_greedy=e_greedy_start,
                                lr=learning_rate,
                                random_seed=random_seed)
        DEQN_list.append(DEQN_tmp)


    # SUs sense the environment and get the sensing result (contains sensing errors)
    env.render(0)
    observation = env.sense(active_sensor, 0)

    t = time.time()
    for step in range(total_episode):

        # SU choose action based on observation
        action = np.zeros(n_su).astype(np.int32)
        for k in range(n_su):
            action[k] = DEQN_list[k].choose_action(observation[k, :])

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

        # Store one episode (s, a, r, s')
        for k in range(n_su):
            state = observation[k, :]
            state_ = observation_[k, :]
            DEQN_list[k].store_transition(state, action[k], reward[k], state_)

        # Each SU learns their DQN model
        if ((step + 1) % (batch_size) == 0):
            for k in range(n_su):
                DEQN_list[k].learn_new(batch_size, step, method='double')
                DEQN_list[k].epsilon = epsilon[k]
                '''
                if (epsilon[k] >= 0.8):
                    DEQN_list[k].update_lr(0.001)
                else:
                    DEQN_list[k].update_lr(0.01)
                '''


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

        # Update epsilon
        if ((step + 1) % epsilon_update_period == 0):
            for k in range(n_su):
                epsilon[k] = min(e_greedy_end, epsilon[k] + e_increase)
                #epsilon[k] = e_greedy_end


        # swap observation
        observation = observation_

    elapsed = time.time() - t
    print('Elapsed time = %.4f sec' % elapsed)
    observation_last = copy.deepcopy(observation)
    env_last = copy.deepcopy(env)

    '''
    for k in range(n_su):
        file = open('DQN_RC_SU%d.obj'%k, 'wb')
        dill.dump(DQN_RC_list[k], file)
        file.close()
    '''

    file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
    model_name = 'DEQN%d' % n_layers
    print(model_name)

    np.save(file_folder + 'reward_SU_' + model_name + '_seed%d_0.3' % random_seed, reward_SU)
    np.save(file_folder + 'elapsedTime_' + model_name + '_seed%d_0.3' % random_seed, elapsed)
