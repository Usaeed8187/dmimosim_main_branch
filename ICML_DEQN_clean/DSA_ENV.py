import numpy as np
import matplotlib.pyplot as plt
from mobility import random_waypoint
from scipy.io import loadmat
from matplotlib import colors as mcolors

from RaminUtils import burstyData # This line was added by Ramin

class DSA_env():
    def __init__(
            self,
            n_channel,
            n_pu,
            n_su,
            channel_bs,
            PUT_BS,
            SUT_BS,
            rayTraEnv,
            punish_interfer_PU = -2,

    ):

        self.n_channel = n_channel  # The number of channels
        self.n_su = n_su            # The number of SUs
        self.n_pu = n_pu            # The number of PUs per channel
        self.channel_bs = channel_bs # The BSs per channel
        self.rayTraEnv = rayTraEnv

        self.area = self.rayTraEnv.area
        self.PUT_BS = PUT_BS
        self.SUT_BS = SUT_BS

        # Time subframe (1ms)
        self.t_subframe = 1

        # total number of period (60000)
        self.total_period = 150000

        # Duration of period (100ms)
        self.t_period = 100

        # Duration of sensing in one pelriod (20ms)
        self.t_sense_period = 20

        # Duration of total time (s)
        self.t_total = int(self.total_period * self.t_period / 1000)

        # Initialize the PU behaviors
        self._load_pu_traffic()

        # Initialize the PU state
        #self.channel_state = np.ones(self.n_channel)
        self.pu_state_period = np.zeros((self.n_pu, self.n_channel, self.t_period))

        self.bps_channel_period = np.zeros((self.n_channel, self.t_period))


        # Transmit power of PU and SU (mW)
        self.PU_power = 400
        self.SU_power = 400

        # Background noise
        #Noise_spectral_dBm = -160 # (dBm/Hz) # -164
        #self.B = 5 # (MHz)
        #Noise_dBm = Noise_spectral_dBm + 10 * np.log10(self.B * (10 ** 6)) # (dBm)
        #self.Noise = 10 ** (Noise_dBm / 10) # (mW)
        self.Noise = 10 ** (-10) # mW

        self._calculate_channel()

        # The punishment for interfering PUs
        self.punish_interfer_PU = punish_interfer_PU

        # The bps threshold for interfering PUs
        self.warning_threshold = 3 # 1.5

        # The bps threshold for SUs' collision
        self.SU_collision_threshold = 2 # 1.5

        # traffic model
        self.packet_size = 8 * 0.5 * (10 ** 3)  # packet size (Kbits)
        self.buffer_pu = np.zeros((self.n_pu, self.n_channel))  # the buffer of each user (Kbits)

        # resource allocation
        PRB_RBG = 4  # the number of PRBs of 1 RBG
        BW_PRB = 180  # bandwidth of 1 PRB (kHz)
        RE_PRB = 168  # the number of REs of 1 PRB during one subframe
        RE_data_PRB = 132  # the number of REs for data of 1 PRB during one subframe
        RE_RBG = RE_PRB * PRB_RBG  # the number of REs of 1 RBG during one subframe
        self.RE_data_RBG = RE_data_PRB * PRB_RBG  # the number of REs for data of 1 RBG during one subframe
        self.n_RBG_cell = 10  # the number of RBG of each cell
        self.transmit_data_pu = 0.000001 * np.ones((self.n_pu, self.n_channel))  # (Kbits)


    def _load_pu_traffic(self):
        filename = 'channel_state_2.mat'
        self.pu_traffic_data = loadmat(filename)['channel_state_2']
        if self.n_pu != 4: # This line was added by Ramin
            self.pu_traffic_data = np.zeros((self.n_pu,60000),dtype=int)
            RandomState = np.random.RandomState(70)
            for i in range(self.n_pu):
                self.pu_traffic_data[i,:] = burstyData(2/3,1/3,3.5,60000,RandomState)


    def _calculate_channel(self):

        self.PUT_loc = np.zeros((2, self.n_channel))
        self.PUR_loc = np.zeros((2, self.n_pu, self.n_channel, self.t_total))
        self.SUT_loc = np.zeros((2, self.n_channel))
        self.SUR_loc = np.zeros((2, self.n_su, self.t_total))

        for c in range(self.n_channel):
            self.PUT_loc[:, c] = self.rayTraEnv.BSlocations[self.PUT_BS[c], :]
            self.SUT_loc[:, c] = self.rayTraEnv.BSlocations[self.SUT_BS[c], :]

        for c in range(self.n_channel):
            area_PU = np.array([250, 400])
            center_PU = np.array([75, 0])
            rw_PU = random_waypoint(self.n_pu, dimensions=(area_PU[0], area_PU[1]), velocity=(0.7, 1.0), wt_max=1.0)
            t = 0
            for xy in rw_PU:
                for k in range(self.n_pu):
                    self.PUR_loc[:, k, c, t] = center_PU + xy[k, :] - np.array([area_PU[0]/2, area_PU[1]/2])

                t += 1
                if t == self.t_total:
                    break

        area_SU = np.array([200, 400])
        center_SU = np.array([-100, 0])
        rw_SU = random_waypoint(self.n_su, dimensions=(area_SU[0], area_SU[1]), velocity=(0.7, 1.0), wt_max=1.0)
        t = 0
        for xy in rw_SU:
            for k in range(self.n_su):
                self.SUR_loc[:, k, t] = center_SU + xy[k, :] - np.array([area_SU[0]/2, area_SU[1]/2])

            t += 1
            if t == self.t_total:
                break

        self._plot_location(0)
        self._plot_location(499)
        # plt.show() # Commentend out by Ramin

        self.H_PUR_PUT = np.zeros((self.n_pu, self.n_channel, self.t_total))
        self.H_SUR_SUT = np.zeros((self.n_su, self.n_su, self.n_channel, self.t_total))
        self.H_SUR_PUT = np.zeros((self.n_su, self.n_channel, self.t_total))
        self.H_PUR_SUT = np.zeros((self.n_pu, self.n_su, self.n_channel, self.t_total))

        for c in range(self.n_channel):
            for t in range(self.t_total):
                #print(t)

                for rx in range(self.n_pu):
                    self.H_PUR_PUT[rx, c, t] = self.rayTraEnv.CalChannelGain(self.PUT_BS[c], self.PUR_loc[:, rx, c, t])
                    for tx in range(self.n_su):
                        self.H_PUR_SUT[rx, tx, c, t] = self.rayTraEnv.CalChannelGain(self.SUT_BS[c], self.PUR_loc[:, rx, c, t])

                for rx in range(self.n_su):
                    for tx in range(self.n_su):
                        self.H_SUR_SUT[rx, tx, c, t] = self.rayTraEnv.CalChannelGain(self.SUT_BS[c], self.SUR_loc[:, rx, t])
                    self.H_SUR_PUT[rx, c, t] = self.rayTraEnv.CalChannelGain(self.PUT_BS[c], self.SUR_loc[:, rx, t])
                    # modified
                    #self.H_SUR_PUT[rx, c, t] = self.rayTraEnv.CalChannelGain(self.PUT_BS[c], self.SUT_loc[:, c])

    def _plot_location(self, period):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_name = ['blue', 'green', 'magenta', 'red', 'gold', 'cyan', 'orange', 'lime', 'pink', 'gray',
                      'purple', 'brown', 'olive', 'darkcyan', 'deeppink']

        # Plot locations of PUs and SUs
        time = int(period * self.t_period / 1000)
        plt.figure(figsize=(6, 6))

        plt.plot(self.PUT_loc[0, 0], self.PUT_loc[1, 0], 'bs', label='PUT')
        plt.plot(self.SUT_loc[0, 0], self.SUT_loc[1, 0], 'rP', label='SUT')
        for c in range(self.n_channel):
            #plt.plot(self.PUR_loc[0, :, c, time], self.PUR_loc[1, :, c, time], 'x', color=colors[color_name[c]], label='PUR (C%d)'%(c+1))
            plt.plot(self.PUR_loc[0, :, c, time], self.PUR_loc[1, :, c, time], 'bx')
        plt.plot(self.PUR_loc[0, :, c, time], self.PUR_loc[1, :, c, time], 'bx', label='PUR')

        plt.plot(self.SUR_loc[0, :, time], self.SUR_loc[1, :, time], 'r^', label='SUR')

        plt.annotate(
            'PBS',
            xy=(self.PUT_loc[0, 0], self.PUT_loc[1, 0]), xytext=(0, 4),
            textcoords='offset points', ha='center', va='bottom',
        )

        plt.annotate(
            'SBS',
            xy=(self.SUT_loc[0, 0], self.SUT_loc[1, 0]), xytext=(0, 4),
            textcoords='offset points', ha='center', va='bottom',
        )

        for n in range(self.n_su):
            labelstr = "SU%d" % (n+1)
            plt.annotate(
                labelstr,
                xy=(self.SUR_loc[0, n, time], self.SUR_loc[1, n, time]), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom',
            )

        for c in range(self.n_channel):
            for n in range(self.n_pu):
                labelstr = "PU%d" % (c*4 + n + 1)
                plt.annotate(
                    labelstr,
                    xy=(self.PUR_loc[0, n, c, time], self.PUR_loc[1, n, c, time]), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom',
                )
        #plt.legend(loc='upper left')
        plt.ylabel('y', fontsize=18)
        plt.xlabel('x', fontsize=18)
        plt.xlim(-self.area[0]/2, self.area[0]/2)
        plt.ylim(-self.area[1]/2, self.area[1]/2)
        #plt.title('Period %d' % (period+1))
        plt.title('Time: %dsec' % (time+1))
        #plt.show()


    def sense(self, active_sensor, period):

        channel_state = np.sum(self.pu_state_period[:, :, :self.t_sense_period], axis=0)
        channel_state = (channel_state > 0).astype(int)

        time = int(period * self.t_period / 1000)

        # Calculate the sensed PU signal at SUR
        sensed_PU_signal = np.sqrt(self.PU_power * self.H_SUR_PUT[:, :, time])
        sensed_PU_signal = np.expand_dims(sensed_PU_signal, axis=2)
        sensed_PU_signal = np.repeat(sensed_PU_signal, self.t_sense_period, axis=2)

        const = 1 * (10 ** 8)  # normalize the sensed signal
        threshold = np.zeros((self.n_su, 1 + self.n_channel))
        # Noise = np.random.normal(0, self.Noise ** 0.5, size=(self.n_su, self.t_sense_period)) # Line commented out by Ramin
        Noise = np.random.normal(0, self.Noise ** 0.5, size=(self.n_su, self.n_channel , self.t_sense_period)) # Line added by Ramin
        for n in range(self.n_su):
            c = np.where(active_sensor[n, :] == 1)[0]
            # sensed_signal = np.multiply(sensed_PU_signal[n, c, :], channel_state[c, :]) + Noise[n, :] # Line commented out by Ramin
            sensed_signal = np.multiply(sensed_PU_signal[n, c, :], channel_state[c, :]) + Noise[n , c , :] # Line modified by Ramin
            sensed_signal_power = sensed_signal ** 2
            sensed_signal_power_sum = np.sum(sensed_signal_power)
            threshold[n, 0] = threshold[n, 0] + sensed_signal_power_sum * const

        sensed_signal_all_channels = np.multiply(sensed_PU_signal[:, :, :], channel_state[np.newaxis,:, :]) + Noise[: ,: , :] # Line added by Ramin to capture sensed signal on all channels
        sensed_signal_power_all_channels = sensed_signal_all_channels ** 2 # Line added by Ramin
        self.sensed_signal_power_sum_all_channels = np.sum(sensed_signal_power_all_channels,axis=-1) # Line added by Ramin
        
        threshold[:, 0] = threshold[:, 0] / self.t_sense_period
        #if np.sum(channel_state[c, :]) > 0:
        #    print(threshold[0, 0])

        max_threshold = 5
        # threshold[:, 0] = min(max_threshold, threshold[:, 0])
        threshold[:, 0][threshold[:, 0]>max_threshold] = max_threshold # This line was added by Ramin
        # The above line was added to support multiple SUs

        # sensed channel indicator
        threshold[:, 1:(self.n_channel+1)] = active_sensor

        return threshold


    def _update_buffer(self, time):

        pu_access = self.pu_traffic_data[:, time]
        pu_access = np.expand_dims(pu_access, axis=1)
        pu_access = np.repeat(pu_access, self.n_channel, axis=1)

        # modify
        #pu_access = np.ones((self.n_pu, self.n_channel))

        for c in range(self.n_channel):
            for k in range(self.n_pu):
                if pu_access[k, c] == 1:
                    self.buffer_pu[k, c] += self.packet_size

    def _scheduling_sense(self):
        # The cell schedules RBGs based on (R**a)/(T**b)
        # The parameters for scheduling
        a = 1
        b = 1

        for c in range(self.n_channel):
            for t in range(self.t_sense_period):
                # The data (bits) of PUs with 1 RBG
                data_RBG_pu = self.RE_data_RBG * self.bps_pu_sense[:, c]
                data_subframe = data_RBG_pu * self.n_RBG_cell
                #served_user_bl = (self.buffer_pu[:, c] > 0) & (self.bps_pu_sense[:, c] > 0)
                served_user_bl = (self.buffer_pu[:, c] > 0)
                served_user = np.where(served_user_bl)[0]

                if served_user.size > 0:
                    metric = (data_subframe[served_user] ** a) / (self.transmit_data_pu[served_user, c] ** b)
                    max_metric_index = np.argmax(metric)
                    selected_pu = served_user[max_metric_index]
                    selected_data_subframe = data_subframe[selected_pu] / 1000
                    buffer_before = self.buffer_pu[selected_pu, c]
                    buffer_after = max(0, self.buffer_pu[selected_pu, c] - selected_data_subframe)
                    self.buffer_pu[selected_pu, c] = buffer_after
                    self.transmit_data_pu[selected_pu, c] += (buffer_before - buffer_after)
                    self.pu_state_period[selected_pu, c, t] = 1
                    self.bps_channel_period[c, t] = self.bps_pu_sense[selected_pu, c]


    def _scheduling(self):
        # The cell schedules RBGs based on (R**a)/(T**b)
        # The parameters for scheduling
        a = 1
        b = 1

        for c in range(self.n_channel):
            for t in range(self.t_sense_period, self.t_period):
                # The data (bits) of PUs with 1 RBG
                data_RBG_pu = self.RE_data_RBG * self.bps_pu[:, c]
                data_subframe = data_RBG_pu * self.n_RBG_cell
                #served_user_bl = (self.buffer_pu[:, c] > 0) & (self.bps_pu[:, c] > 0)
                served_user_bl = (self.buffer_pu[:, c] > 0)
                served_user = np.where(served_user_bl)[0]

                if served_user.size > 0:
                    metric = (data_subframe[served_user] ** a) / (self.transmit_data_pu[served_user, c] ** b)
                    max_metric_index = np.argmax(metric)
                    selected_pu = served_user[max_metric_index]
                    selected_data_subframe = data_subframe[selected_pu] / 1000
                    buffer_before = self.buffer_pu[selected_pu, c]
                    buffer_after = max(0, self.buffer_pu[selected_pu, c] - selected_data_subframe)
                    self.buffer_pu[selected_pu, c] = buffer_after
                    self.transmit_data_pu[selected_pu, c] += (buffer_before - buffer_after)
                    self.pu_state_period[selected_pu, c, t] = 1
                    self.bps_channel_period[c, t] = self.bps_pu[selected_pu, c]

    def _quantize_reward(self, bps):

        if (bps < 1):
            reward = 0
        elif (bps >= 1 and bps < 2):
            reward = 1
        elif (bps >= 2 and bps < 3):
            reward = 2
        else:
            reward = 3

        return reward

    def render(self, period):
        # Update PU buffers
        if period * self.t_period % 1000 == 0:
            time = int(period * self.t_period / 1000)
            self._update_buffer(time)

        self._calculate_bps_pu_sense(period)
        self._scheduling_sense()

    def access(self, action, period):

        self.success_SU = np.zeros(self.n_su)  # a SU doesn't collide with other SUs or degrades PU data rate
        self.fail_channel = np.zeros(self.n_channel)  # a SU collides with a PU and degrades PU data rate
        self.success_channel = np.zeros(self.n_channel)
        self.fail_SU = np.zeros(self.n_su)  # a SU collides with other SUs
        self.access_SU = np.zeros(self.n_su)  # the number of SUs' access

        self.reward = - np.ones(self.n_su)
        self.bps_achieved_SU = np.zeros(self.n_su)
        self.bps_achieved_channel = np.zeros(self.n_channel)

        self._calculate_bps_pu(action, period)
        self._scheduling()
        self._calculate_bps_su(action, period)
        # (TODO) self._scheduling_su()
        self._calculate_reward(action)

        # Initialize
        self.pu_state_period = np.zeros((self.n_pu, self.n_channel, self.t_period))
        self.bps_channel_period = np.zeros((self.n_channel, self.t_period))

        return self.reward

    def _calculate_bps_pu(self, action, period):

        time = int(period * self.t_period / 1000)
        self.bps_pu = np.zeros((self.n_pu, self.n_channel))

        # Calculate bps of PU
        for c in range(self.n_channel):
            for n in range(self.n_pu):
                # Calculate the interference between PUR/SUT
                Interferecne_PUR_SUT = 0
                interfered_SUT = np.where(action == 2 * c)[0]
                for m in interfered_SUT:
                    Interferecne_PUR_SUT += self.H_PUR_SUT[n, m, c, time] * self.SU_power

                SINR = self.H_PUR_PUT[n, c, time] * self.PU_power / (Interferecne_PUR_SUT + self.Noise)
                self.bps_pu[n, c] = self._SINR_to_bps(SINR)

    def _calculate_bps_pu_sense(self, period):

        time = int(period * self.t_period / 1000)

        self.bps_pu_sense = np.zeros((self.n_pu, self.n_channel))

        # Calculate bps of PU
        for c in range(self.n_channel):
            for n in range(self.n_pu):
                SINR = self.H_PUR_PUT[n, c, time] * self.PU_power / self.Noise
                self.bps_pu_sense[n, c] = self._SINR_to_bps(SINR)


    def _calculate_bps_su(self, action, period):

        channel_state = np.sum(self.pu_state_period[:, :, self.t_sense_period: self.t_period], axis=0)
        if np.amax(channel_state) > 1:
            print('error')
        channel_state = (channel_state > 0).astype(int)

        time = int(period * self.t_period / 1000)

        '''
        pu_access = self.pu_traffic_data[:, time]
        print('time: ', time, '  period: ', period)
        print('PU bps: ', self.bps_pu[:, 0])
        print('PU access: ', pu_access)
        print('SU access: ', (action[0] % 2) == 0)
        print('channel access rate: ', np.sum(channel_state) / (self.t_period - self.t_sense_period))
        '''
        '''
        pu_access = self.pu_traffic_data[:, time]
        print('time: ', time, '  period: ', period)
        print('PU access: ', pu_access)
        print('channel access rate: ', np.sum(channel_state) / (self.t_period - self.t_sense_period))
        '''

        self.bps_su = np.zeros((self.n_su, self.t_period - self.t_sense_period))

        # Calculate bps of SU
        for k in range(self.n_su):
            access_channel = int(action[k] / 2)

            '''
            # Calculate the interference between SUR/SUT
            Interferecne_SUR_SUT = 0
            interfered_SUT = np.where(action == action[k])[0]
            interfered_SUT = interfered_SUT[interfered_SUT != k]
            for m in interfered_SUT:
                Interferecne_SUR_SUT += self.H_SUR_SUT[k, m, access_channel, time] * self.SU_power
            '''

            # Calculate the interference between SUR/PUT
            Interferecne_SUR_PUT = np.zeros(self.t_period - self.t_sense_period)
            for t in range(self.t_period - self.t_sense_period):
                if channel_state[access_channel, t] == 1:
                    Interferecne_SUR_PUT[t] = self.H_SUR_PUT[k, access_channel, time] * self.PU_power


            # Calculate the total interference
            #Interferecne = Interferecne_SUR_SUT + Interferecne_SUR_PUT
            Interferecne = Interferecne_SUR_PUT

            SINR = self.H_SUR_SUT[k, k, access_channel, time] * self.SU_power / (Interferecne + self.Noise)

            for t in range(self.t_period - self.t_sense_period):
                self.bps_su[k, t] = self._SINR_to_bps(SINR[t])

        #print(np.mean(self.bps_su, axis=1))

    def _calculate_reward(self, action):
        # Calculate the reward of SU
        pu_state = self.pu_state_period[:, :, self.t_sense_period: self.t_period]
        bps_channel = self.bps_channel_period[:, self.t_sense_period: self.t_period]
        channel_state = np.sum(self.pu_state_period[:, :, self.t_sense_period: self.t_period], axis=0)
        channel_state = (channel_state > 0).astype(int)

        for k in range(self.n_su):
            if ((action[k] % 2) == 1):  # action is not accessing any channel
                self.access_SU[k] = 0
            else:  # action is accessing one channel
                access_channel = int(action[k] / 2)
                self.access_SU[k] = 1

        for c in range(self.n_channel):
            if np.sum(channel_state[c, :]) > 0:
                # Check if SU collides with PU and causes strong interference
                if np.sum(action == 2 * c) > 0:
                    average_bps_channel = np.sum(bps_channel[c, :]) \
                                          / np.sum(channel_state[c, :])
                    if average_bps_channel < self.warning_threshold:
                        # collision with PU
                        self.fail_channel[c] = 1
                        self.fail_SU[k] = 1
                    else:
                        self.fail_channel[c] = 0
                        self.success_channel[c] = 1
                else:
                    self.success_channel[c] = 1


        for k in range(self.n_su):
            if ((action[k] % 2) == 1):  # action is not accessing any channel
                self.reward[k] = 0
            else:  # action is accessing one channel
                access_channel = int(action[k] / 2)
                # Check if SU collides with PU and causes strong interference
                if np.sum(channel_state[access_channel, :]) > 0:
                    average_bps_channel = np.sum(bps_channel[access_channel, :]) \
                                          / np.sum(channel_state[access_channel, :])
                    if average_bps_channel < self.warning_threshold:
                        # collision with PU
                        self.reward[k] = -1
                    else:
                        self.success_SU[k] = 1
                        self.reward[k] = +1
                else:
                    self.success_SU[k] = 1
                    self.reward[k] = +1

    def _SINR_to_bps(self, SINR):

        bps = 0
        SINR_dB = 10 * np.log10(SINR)

        SINR_dB_level = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7,
                         14.1, 16.3, 18.7, 21.0, 22.7]

        bps_level = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
                     3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

        for i in range(15):
            if (i != 14):
                if SINR_dB >= SINR_dB_level[i] and SINR_dB < SINR_dB_level[i+1]:
                    bps = bps_level[i]
                    break
            else:
                if SINR_dB >= SINR_dB_level[i]:
                    bps = bps_level[i]

        return bps

    def render_sensor(self, action):

        active_sensor = np.zeros((self.n_su, self.n_channel)).astype(np.int32)
        initial_sensed_channel = np.floor(action / 2).astype(np.int32)
        for k in range(self.n_su):
            active_sensor[k, initial_sensed_channel[k]] = 1
        return active_sensor