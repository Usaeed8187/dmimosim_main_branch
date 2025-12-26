from DSA_ENV import DSA_env
from RayTraModel import *
import tensorflow as tf
from DQN_MLP import MLP1
import numpy as np
import copy
import dill
import time


if __name__ == "__main__":

    random_seed = 1

    # Number of channel, Number of PUs per channel, Number of SUs
    n_channel = 3
    n_pu = 4
    n_su = 1


    concatenate = 5
    dim_actions = n_channel * 2  # The action space size
    dim_states = (n_channel + 1) # The sensing result space
    dim_states_concatenate = dim_states * concatenate

    # training parameters
    batch_size = 500  # 300
    total_episode = batch_size * 300  # 200
    epsilon_update_period = batch_size * 1
    e_greedy_start = 0.7  # 0.5
    e_greedy_end = 0.9  # 0.9
    e_increase = (e_greedy_end - e_greedy_start) / 290  # 190
    learning_rate = 0.001  # 0.01
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

    # Load the initial environment
    file = open('env_PU%d_SU%d_seed%d.obj' % (n_pu, n_su, random_seed), 'rb')
    env = dill.load(file)
    file.close()


    random_seed = 15
    np.random.seed(random_seed)



    # Initialize the sensor
    active_sensor = np.zeros((n_su, n_channel)).astype(np.int32)
    initial_sensed_channel = np.random.choice(n_channel, n_su)
    for k in range(n_su):
        active_sensor[k, initial_sensed_channel[k]] = 1

    # Initialize the DRQN
    DQN_list = []


    for k in range(n_su):
        DQN_tmp = MLP1(dim_actions, dim_states_concatenate,
                        reward_decay=0.9,
                        e_greedy=e_greedy_start,
                        memory_size=batch_size,
                        lr=learning_rate,
                        random_seed=random_seed
                        )
        DQN_list.append(DQN_tmp)

    # SUs sense the environment and get the sensing result (contains sensing errors)
    env.render(0)
    observation_input = np.zeros((1, dim_states_concatenate))
    observation = env.sense(active_sensor, 0)
    observation_input[0, :dim_states] = observation[0, :]

    t = time.time()
    for step in range(total_episode):

        # SU choose action based on observation
        action = np.zeros(n_su).astype(np.int32)
        # SU choose action based on observation
        for k in range(n_su):
            action[k] = DQN_list[k].choose_action(observation_input[k, :])

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
            observation_input_ = np.zeros((1, dim_states_concatenate))
            observation_input_[0, dim_states:] = observation_input[0, :(dim_states_concatenate - dim_states)]
            observation_input_[0, :dim_states] = observation_[0, :]

        # Store one episode (s, a, r, s')
        for k in range(n_su):
            state = observation_input[k, :]
            state_ = observation_input_[k, :]
            DQN_list[k].store_transition(state, action[k], reward[k], state_)

        # Each SU learns their DQN model
        if ((step + 1) % (batch_size) == 0):
            for k in range(n_su):
                DQN_list[k].learn()
                DQN_list[k].epsilon = epsilon[k]


        if ((step+1) % batch_size == 0):
            index = np.arange(step+1-batch_size, step+1)
            print('Training episode = %d' % int((step + 1) / batch_size))
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
                # epsilon[k] = e_greedy_end

        # swap observation
        observation_input = observation_input_

    elapsed = time.time() - t
    print('Elapsed time = %.4f sec' % elapsed)

    observation_last = copy.deepcopy(observation)
    env_last = copy.deepcopy(env)


    file_folder = './result/PU%d_SU%d_C%d/' % (n_pu, n_su, n_channel)
    model_name = 'DQN'
    np.save(file_folder + 'reward_SU_' + model_name + '_seed%d' % random_seed, reward_SU)
    np.save(file_folder + 'elapsedTime_' + model_name + '_seed%d' % random_seed, elapsed)

    tf.reset_default_graph()