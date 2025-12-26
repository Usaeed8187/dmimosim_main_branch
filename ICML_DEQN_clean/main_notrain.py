from DSA_Period_env import DSA_Period
import matplotlib.pyplot as plt
import dill
import numpy as np
import copy


if __name__ == "__main__":

    n_channel = 4
    n_su = 6

    dim_actions = n_channel * 2  # The action space size
    dim_states = n_channel + 1  # The sensing result space

    # training parameters
    batch_size = 300
    total_episode = batch_size * 200


    # Initialize some record values
    dataRate_SU = np.zeros((n_su, total_episode))
    dataRate_PU = np.zeros((n_channel, total_episode))


    # Initialize the environment
    env = DSA_Period(n_channel, n_su)

    # Store the initial environment
    file = open('env_PU%d_SU%d.obj' % (n_channel, n_su), 'wb')
    dill.dump(env, file)
    file.close()


    for step in range(total_episode):

        # SU choose action based on observation
        action = np.zeros(n_su).astype(np.int32)
        for k in range(n_su):
            action[k] = 1

        # SU take action and get the reward
        reward = env.access(action, step)

        # Record values
        dataRate_SU[:, step] = env.dataRate_SU
        dataRate_PU[:, step] = env.dataRate_PU

        # update the PU states
        env.render_PU_state()


        if ((step+1) % batch_size == 0):
            index = np.arange(step+1-batch_size, step+1)
            print('Training time = %d' % (step + 1))



    file_folder = './result/PU%d_SU%d/' % (n_channel, n_su)
    model_name = 'NoTrain'
    print(model_name)


    np.save(file_folder + 'dataRate_SU_' + model_name, dataRate_SU)
    np.save(file_folder + 'dataRate_PU_' + model_name, dataRate_PU)

