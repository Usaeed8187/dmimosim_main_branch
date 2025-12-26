import numpy as np
from DQN_RC_new import DeepQNetwork

def generate_DQNs(DQN: DeepQNetwork, list_of_spectral_radii:list , random_seed):
    # assert (DQN.memory_counter==DQN.memory_size),'Memory counter should be pointing at the end of the memory'
    list_of_DQNs = [] #[DQN]
    for i,radius in enumerate(list_of_spectral_radii):
        new_DQN = (DeepQNetwork(n_actions=DQN.n_actions,
                                        n_features=DQN.n_features,
                                        memory_size=DQN.memory_size,
                                        n_layers=DQN.n_layers,
                                        reward_decay=DQN.gamma,
                                        e_greedy=DQN.epsilon,
                                        lr=DQN.lr,
                                        random_seed=random_seed,
                                        spectral_radius=radius))
        # for j in range(len(new_DQN.memory)):
        #     experience = new_DQN.memory[j,:]
        #     state = experience[:new_DQN.n_features]
        #     new_DQN.eval_net.predict(state,0)
        list_of_DQNs.append(new_DQN)

    return list_of_DQNs


def BiasError(DQN: DeepQNetwork, learning_method='double'):
    n_FQIiters = DQN.training_iteration // DQN.replace_target_iter
    n_samples = DQN.training_batch_size
    targ_net_next_q_vals = DQN.target_net.predict_training(index=np.arange(1,DQN.memory_size))
    main_net_next_q_vals = DQN.eval_net.predict_training(index=np.arange(1,DQN.memory_size))
    # DQN.memory is of shape [None, DQN.n_features + 1 + 1 + DQN.n_features],
    # where the first +1 refers to action and the second +1 referes to reward
    episode_actions = DQN.memory[:-1,DQN.n_features-1+1]#:DQN.n_features+1]
    episode_rewards = DQN.memory[:-1,DQN.n_features-1+2]#:DQN.n_features+2]
    target_q_value = np.max(targ_net_next_q_vals,axis=1)
    pass
    if learning_method=='double':
        # we have to get the Q values of the target network, corresponding to the action that the main net suggests
        target_q_values = episode_rewards+ DQN.gamma*\
                            targ_net_next_q_vals[np.arange(targ_net_next_q_vals.shape[0]), # all rows
                                                np.argmax(main_net_next_q_vals,axis=1)] # the columns with the max Q value
        pass
    elif learning_method == 'normal':
        #Exception has occurred: ValueError
        #operands could not be broadcast together with shapes (499,) (499,6) 
        update_target = episode_rewards + DQN.gamma*np.max(targ_net_next_q_vals,axis=1)
    else: raise ValueError(f'"{learning_method}" is not a valid learning method.')
    # action_based_indices = []
    BiasError = 0
    for action in range(DQN.n_actions):
        action_based_indices = (episode_actions==action) # indices of states which led to
        # action "action"
        X = (DQN.eval_net.extended_states[:-1,:][action_based_indices,:]).T # This 
        # is the main DEQN vectors right before W_out, which had led to action "action"
        # The vectors are stacked together so the shape is (d_h, number of vectors that 
        # led to action "action").
        A = 2 * X @ X.T # The A matrix in (0.5*wᵀAw + wᵀb). Is of shape (d_h,d_h)
        action_specific_targets = target_q_values[action_based_indices].reshape([-1,1]) # of shape
        # (number of vectors that led to action "action", 1)
        b = - 2 * X @ action_specific_targets # The b vector in (0.5*wᵀAw + wᵀb). Is of shape (d_h,1)
        try:
            # new_weights = np.linalg.solve(A,-b)
            etta = 0.1
            new_weights = np.linalg.solve(A+ etta*np.eye(A.shape[0]),-b)
        except np.linalg.LinAlgError:
            if A.shape[0] == A.shape[1]:
                raise Exception(f'The A matrix which is of shape {A.shape} is singular')
            else:
                raise Exception(f'The A matrix which is of shape {A.shape} is not a square matrix')
        # DQN.eval_net.W_out = 
        # error_for_this_action = (0.5 * b.T @ new_weights + action_specific_targets.T @ action_specific_targets)
        error_for_this_action = 0.5 * new_weights.T @ A @ new_weights + new_weights.T @ b + action_specific_targets.T @ action_specific_targets
        # print(error_for_this_action)
        if error_for_this_action< 0 :
            print('A negative error occured!')
            2*A
        BiasError = BiasError +  error_for_this_action
        pass
    print(DQN.spectral_radius,'radius, bias=',BiasError)
    return BiasError
    #

def VarianceError(DQN:DeepQNetwork,Vmax,epsilon):
    n_FQIiters = DQN.training_iteration // DQN.replace_target_iter
    n_samples = DQN.training_batch_size
    T = DQN.memory_size
    C1 = 0.0001*(8*np.sqrt(2*T)+256/Vmax)#*(Vmax**2 / T)
    C2 = 0.0001*(16 + 4*np.sqrt(2*n_samples) + 36)#*Vmax*epsilon
    arr_of_states = DQN.memory[:,:DQN.n_features]
    # Cov_UB = np.zeros(DQN.memory_size-1)
    # for t in range(DQN.memory_size-1):
    #     B_in = 
    #     Cov_UB = 0
    B_in = np.linalg.norm(DQN.eval_net.W_in[0],2)
    B_out = np.linalg.norm(DQN.eval_net.W_out,2)
    B_rec = np.linalg.norm(DQN.eval_net.W[0],2)
    B_X = np.max(np.linalg.norm(arr_of_states,axis=1))
    d_y = DQN.n_actions
    d_h = DQN.eval_net.W[0].shape[0]
    log_covering_num = (B_in*B_out*B_X)**2*d_y*np.log(2*d_h*d_y)*\
        ((1-B_rec**T)/(1-B_rec))**2 / epsilon**2
    variance = C1*Vmax**2/T * log_covering_num + C2*Vmax*epsilon
    print(DQN.spectral_radius,'radius, cover=',log_covering_num)
    # print('B_X',B_X)
    # print('B_in',B_in)
    # print('B_out',B_out)
    # print('B_rec',B_rec)
    return variance
