import numpy as np

class MyopicAgent():
    def __init__(self,num_channels,p_b2g=None,p_g2g=None,
                 reward_idle = 0, reward_tx_success = 1 , reward_tx_fail = -1 , 
                 num_steps_for_estimation = 2000) -> None:
        '''
        if p_b2g and p_g2g are None, Empirical Myopic is activated.
        '''
        self.num_channels = num_channels
        assert (((p_b2g is None) and (p_g2g is None)) or ((p_b2g is not None) and (p_g2g is not None))), 'Both probs have to be either None or both have to have values.'
        if (p_b2g is None) or (p_g2g is None):
            print('Activating the Empricial Myopic strategy...')
            self.estimate_probs = True
            self.b2bCount = np.zeros(num_channels)
            self.b2gCount = np.zeros(num_channels)
            self.g2bCount = np.zeros(num_channels)
            self.g2gCount = np.zeros(num_channels)
            self.est_count = 0
            self.done_estimation = False
        else:
            print('Activating the regular Myopic strategy...')
            self.estimate_probs = False
            assert np.shape(p_b2g) == (num_channels,)
            assert np.shape(p_g2g) == (num_channels,)
            self.p_b2g = p_b2g
            self.p_g2g = p_g2g
        self.reward_idle = reward_idle
        self.reward_tx_success = reward_tx_success
        self.reward_tx_fail = reward_tx_fail
        self.num_est_steps = num_steps_for_estimation

        self.belief = 0.5 * np.ones(num_channels) # probabibility of being in good state at time t  
        pass

    def store_observation(self, state_curr:np.ndarray, state_next:np.ndarray):
        '''
        state_curr and state_next must be vectors of size num_channels
        that include 0's and 1's only. 0 representing "bad" and 1
        representing "good".
        '''
        assert self.estimate_probs, 'This object was not instantiated to estimate the probs.'
        assert np.shape(state_curr) == (self.num_channels,)
        assert np.shape(state_next) == (self.num_channels,)
        assert (state_curr.dtype == int) and (state_next.dtype == int)
        # assert (0<=state_curr<=1).all() and (0<=state_next<=1).all()
        assert ((state_curr<=1) & (state_curr>=0) & (state_next<=1) & (state_next>=0)).all()
        
        difference = state_next - state_curr
        # if difference is -1, it means we have moved from good to bad
        # if difference is +1, it means we have moved from bad to good
        # if differnece is 0, it could be either good to good or bad to bad
        self.b2gCount += (difference == -1)
        self.g2bCount += (difference == +1)
        self.g2gCount += ((difference == 0) & (state_curr==1))
        self.b2bCount += ((difference == 0) & (state_curr==0))
        self.est_count += 1

        if self.est_count == self.num_est_steps:
            self.done_estimation = True
            self.p_b2g = self.b2gCount / self.est_count
            self.p_g2g = self.g2gCount / self.est_count
        pass

    def choose_action(self,state:np.ndarray,power_threshold):
        '''
        If the last element of state, which is the power received, is less
        than power_threshold, then it is assumed that we are in a good state.
        Otherwise, we are in a bad state.
        '''
        assert np.shape(state) == (self.num_channels+1,)
        if self.estimate_probs: assert self.done_estimation, 'Cannot choose an action before etimation of probs has finished.'
        chosen_channel = (state[1:]==1)
        state_is_good = (state[1] <= power_threshold)
        self.belief[chosen_channel] = (self.p_g2g[chosen_channel] if state_is_good else self.p_b2g[chosen_channel])
        self.belief[~chosen_channel] = (self.belief[~chosen_channel]*self.p_g2g[~chosen_channel] +
                                        (1-self.belief[~chosen_channel])*self.p_b2g[~chosen_channel])
        
        ## I'm expexting this to hold:
        assert ((0<=self.belief)&(self.belief<=1)).all()

        # The chosen channel to sense is the one believed to be good with most probability
        chosen_channel_next = np.zeros(self.num_channels).astype(bool)
        chosen_channel_next[np.argmax(self.belief)] = True
        
        # Compute the expected reward
        expected_reward = (self.belief[chosen_channel_next]*(self.reward_tx_success) + 
                           (1-self.belief[chosen_channel_next])*(self.reward_tx_fail))
        if expected_reward <= 0:
            idle = True
            # action = np.concatenate((chosen_channel_next,np.zeros(self.num_channels)),axis=0) # This is wrong
            action = np.argmax(self.belief)*2 + 1
        else:
            idle = False
            # action = np.concatenate((np.zeros(self.num_channels),chosen_channel_next),axis=0) # This is wrong
            action = np.argmax(self.belief)*2

        return action

