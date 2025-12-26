import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

class RayTraModel:
    def __init__(self, site):

        # load BS/UE locations
        file_folder = './RaytracingData/'
        UElocations = np.load(file_folder + 'UElocations_' + site + '.npy')
        BSlocations = np.load(file_folder + 'BSlocations_' + site + '.npy')

        self.UElocations = UElocations[:, 0:2]
        self.UEheights = UElocations[:, 2]
        self.BSlocations = BSlocations[:, 0:2]
        self.BSheights = BSlocations[:, 2]
        self.numCells = BSlocations.shape[0]
        self.numUEs = UElocations.shape[0]



        # Adjust locations based on the center of all UEs
        UE_center = np.mean(self.UElocations, axis=0)
        self.BSlocations[:, 0] -= UE_center[0]
        self.BSlocations[:, 1] -= UE_center[1]
        self.UElocations[:, 0] -= UE_center[0]
        self.UElocations[:, 1] -= UE_center[1]

        self.area = np.zeros(2)
        self.area[0] = np.amax(self.UElocations[:, 0]) - np.amin(self.UElocations[:, 0])
        self.area[1] = np.amax(self.UElocations[:, 1]) - np.amin(self.UElocations[:, 1])

        # load receivedPower
        self.receivePower_allTilt = np.load(file_folder + 'receivePower_' + site + '.npy')


        self.numTilt = self.receivePower_allTilt.shape[0]
        self.tilt = np.zeros((self.numCells), dtype=int)

        self.receivePower_cell = np.zeros((self.numCells, self.numUEs))
        for cell in range(self.numCells):
            self.receivePower_cell[cell, :] = self.receivePower_allTilt[self.tilt[cell], cell, :]

    def reduceData(self, center, area, selected_cell):
        # Reduce the data based on the selected cells
        # Adjust UE locations based on the center
        # Reduce UE data based on the simulation area

        # Reduce the data based on the selected cells
        self.BSlocations = self.BSlocations[selected_cell,:]
        self.BSheights = self.BSheights[selected_cell]
        self.receivePower_cell = self.receivePower_cell[selected_cell, :]
        self.receivePower_allTilt = self.receivePower_allTilt[:, selected_cell, :]
        self.numCells = selected_cell.size
        self.tilt = self.tilt[selected_cell]

        # Adjust locations based on the center of simulation area
        self.BSlocations[:, 0] -= center[0]
        self.BSlocations[:, 1] -= center[1]
        self.UElocations[:, 0] -= center[0]
        self.UElocations[:, 1] -= center[1]

        height = 1.5
        height_index = np.where(self.UEheights == height)[0]
        self.UElocations = self.UElocations[height_index, :]
        self.UEheights = self.UEheights[height_index]
        self.receivePower_cell = self.receivePower_cell[:, height_index]
        self.receivePower_allTilt = self.receivePower_allTilt[:, :, height_index]

        # delete UE data that are out of simulation area
        self.area = area
        x_min_index = np.where(self.UElocations[:, 0] >= -area[0] / 2)[0]
        self.UElocations = self.UElocations[x_min_index, :]
        self.UEheights = self.UEheights[x_min_index]
        self.receivePower_cell = self.receivePower_cell[:, x_min_index]
        self.receivePower_allTilt = self.receivePower_allTilt[:, :, x_min_index]

        x_max_index = np.where(self.UElocations[:, 0] <= area[0] / 2)[0]
        self.UElocations = self.UElocations[x_max_index, :]
        self.UEheights = self.UEheights[x_max_index]
        self.receivePower_cell = self.receivePower_cell[:, x_max_index]
        self.receivePower_allTilt = self.receivePower_allTilt[:, :, x_max_index]

        y_min_index = np.where(self.UElocations[:, 1] >= -area[1] / 2)[0]
        self.UElocations = self.UElocations[y_min_index, :]
        self.UEheights = self.UEheights[y_min_index]
        self.receivePower_cell = self.receivePower_cell[:, y_min_index]
        self.receivePower_allTilt = self.receivePower_allTilt[:, :, y_min_index]

        y_max_index = np.where(self.UElocations[:, 1] <= area[1] / 2)[0]
        self.UElocations = self.UElocations[y_max_index, :]
        self.UEheights = self.UEheights[y_max_index]
        self.receivePower_cell = self.receivePower_cell[:, y_max_index]
        self.receivePower_allTilt = self.receivePower_allTilt[:, :, y_max_index]

        self.numUEs = self.UElocations.shape[0]

        # Construct a grid
        self.grid_dis = 8
        area_ref_loc = np.array([-area[0] / 2, -area[1] / 2])
        x_range = np.floor(area[0] / self.grid_dis).astype('int')
        y_range = np.floor(area[1] / self.grid_dis).astype('int')
        self.grid = np.zeros((x_range, y_range))

        UE_xindex = np.floor((self.UElocations[:, 0] - area_ref_loc[0]) / self.grid_dis).astype('int')
        UE_yindex = np.floor((self.UElocations[:, 1] - area_ref_loc[1]) / self.grid_dis).astype('int')

        for user in range(self.numUEs):
            self.grid[UE_xindex[user], UE_yindex[user]] = user

        self.ref_loc = np.zeros(2)
        user = 1
        self.ref_loc[0] = self.UElocations[user, 0] - UE_xindex[user] * self.grid_dis
        self.ref_loc[1] = self.UElocations[user, 1] - UE_yindex[user] * self.grid_dis

    def SetTilt(self, tilt, bs_cell):
        # Set the tilt for each cell and recalculate the received power
        self.numBSs = bs_cell.shape[0]


        self.tilt = tilt
        self.receivePower_cell = np.zeros((self.numCells, self.numUEs))
        for bs in range(self.numBSs):
            for k in range(bs_cell.shape[1]):
                cell = bs_cell[bs, k]
                self.receivePower_cell[cell, :] = self.receivePower_allTilt[self.tilt[bs], cell, :]

        self.receivePower = np.zeros((self.numBSs, self.numUEs))
        BSlocations = np.zeros((self.numBSs, 2))
        for bs in range(self.numBSs):
            BSlocations[bs, :] = self.BSlocations[bs_cell[bs, 0], :]
            self.receivePower[bs, :] = np.amax(self.receivePower_cell[bs_cell[bs, :], :], axis=0)
        self.BSlocations = BSlocations

    def show_cellRange(self):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_name = ['blue', 'green', 'magenta', 'red', 'gold', 'cyan', 'orange', 'lime', 'pink', 'gray',
                     'purple', 'brown', 'olive', 'darkcyan', 'deeppink']

        # Use the maximum received power as the served cell for each user
        served_cell = np.argmax(self.receivePower_cell, axis=0)

        plt.figure(figsize=(8 * self.area[0] / self.area[1], 8))

        for cell in range(self.numCells):
            labelstr = 'Cell %d' % (cell + 1)
            served_user = (served_cell == cell)
            plt.plot(self.UElocations[served_user, 0], self.UElocations[served_user, 1], '.',
                     color=colors[color_name[cell]], label=labelstr)
            plt.plot(self.BSlocations[cell, 0], self.BSlocations[cell, 1], 's', color=colors[color_name[cell]])

            plt.annotate(labelstr, xy=(self.BSlocations[cell, 0], self.BSlocations[cell, 1]),
                         xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

        plt.legend(loc='upper right')

        #plt.show()

    def show_BSRange_SINR(self, channel_bs):

        eps1 = 10 ** (-8)
        eps2 = 10 ** (-10)

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_name = ['blue', 'green', 'magenta', 'red', 'gold', 'cyan', 'orange', 'lime', 'pink', 'gray',
                      'purple', 'brown', 'olive', 'darkcyan', 'deeppink']

        n_channel = channel_bs.shape[0]
        for c in range(n_channel):
            receivePower = self.receivePower[channel_bs[c, :], :]
            receivePower = receivePower + eps1

            # Use the maximum received power as the served cell for each user
            served_bs = np.argmax(receivePower, axis=0)
            served_bs[:] = 1 # modify

            plt.figure()

            plt.subplot(1, 2, 1)

            for k in range(channel_bs.shape[1]):
                bs = channel_bs[c, k]
                labelstr = 'BS %d' % (bs + 1)

                served_user = (served_bs == k)
                plt.plot(self.UElocations[served_user, 0], self.UElocations[served_user, 1], '.',
                         color=colors[color_name[bs]], label=labelstr)
                plt.plot(self.BSlocations[bs, 0], self.BSlocations[bs, 1], 's', color=colors[color_name[bs]])

                plt.annotate(labelstr, xy=(self.BSlocations[bs, 0], self.BSlocations[bs, 1]),
                             xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

            plt.legend()
            plt.title('Channel %d' % (c + 1))

            plt.subplot(1, 2, 2)

            max_power = np.amax(receivePower, axis=0)
            max_power = receivePower[1, :]
            interference = np.sum(receivePower, axis=0) - max_power
            SINR = (max_power) / (interference + eps2)
            SINR_dB = 10 * np.log10(SINR)

            bps = np.zeros(self.numUEs)
            for k in range(self.numUEs):
                bps[k] = self._calculate_bps(SINR_dB[k])

            # print(np.sum(bps >= 3))

            for k in range(channel_bs.shape[1]):
                bs = channel_bs[c, k]
                labelstr = 'BS %d' % (bs + 1)

                if k == 0:
                    labelstr = 'SBS'
                else:
                    labelstr = 'PBS'

                plt.plot(self.BSlocations[bs, 0], self.BSlocations[bs, 1], 's', color='k', zorder=10)

                plt.annotate(labelstr, xy=(self.BSlocations[bs, 0], self.BSlocations[bs, 1]), weight='bold',
                             xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

            selected_UE = bps < 1
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='spectral-efficiency < 1')
            selected_UE = (bps >= 1) & (bps < 2)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='1 < spectral-efficiency < 2')
            selected_UE = (bps >= 2) & (bps < 3)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='2 < spectral-efficiency < 3')
            selected_UE = (bps >= 3)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='spectral-efficiency > 3')

            plt.legend(framealpha=1)
            plt.title('Channel %d' % (c + 1))

        #plt.show()

    def show_channel_gain(self, channel_bs):

        eps1 = 10 ** (-8)
        eps2 = 10 ** (-10)

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_name = ['blue', 'green', 'magenta', 'red', 'gold', 'cyan', 'orange', 'lime', 'pink', 'gray',
                      'purple', 'brown', 'olive', 'darkcyan', 'deeppink']

        n_channel = channel_bs.shape[0]
        for c in range(n_channel):
            receivePower = self.receivePower[channel_bs[c, :], :]
            receivePower = receivePower + eps1

            # Use the maximum received power as the served cell for each user
            served_bs = np.argmax(receivePower, axis=0)

            plt.figure()

            plt.subplot(1, 2, 1)

            plt.plot(self.BSlocations[0, 0], self.BSlocations[0, 1], 's', color='k')
            plt.annotate('SBS', xy=(self.BSlocations[0, 0], self.BSlocations[0, 1]),
                         xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)
            plt.plot(self.BSlocations[3, 0], self.BSlocations[3, 1], 's', color='k')
            plt.annotate('PBS', xy=(self.BSlocations[3, 0], self.BSlocations[3, 1]),
                         xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

            PBS_power = receivePower[1, :]


            level = [10**(-8), 10**(-7), 10**(-6)]
            selected_UE = PBS_power < level[0]
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='bps < 1')
            selected_UE = (PBS_power >= level[0]) & (PBS_power < level[1])
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='1 < bps < 2')
            selected_UE = (PBS_power >= level[1]) & (PBS_power < level[2])
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='2 < bps < 3')
            selected_UE = (PBS_power >= level[2])
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='bps > 3')

            plt.legend()
            plt.title('Channel %d' % (c + 1))

    def show_BSRange(self, channel_bs):
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        color_name = ['blue', 'green', 'magenta', 'red', 'gold', 'cyan', 'orange', 'lime', 'pink', 'gray',
                     'purple', 'brown', 'olive', 'darkcyan', 'deeppink']

        n_channel = channel_bs.shape[0]
        for c in range(n_channel):
            receivePower = self.receivePower[channel_bs[c, :], :]

            # Use the maximum received power as the served cell for each user
            served_bs = np.argmax(receivePower, axis=0)

            plt.figure(figsize=(7 * self.area[0] / self.area[1], 7))

            for k in range(channel_bs.shape[1]):
                bs = channel_bs[c, k]
                labelstr = 'BS %d' % (bs + 1)

                served_user = (served_bs == k)
                plt.plot(self.UElocations[served_user, 0], self.UElocations[served_user, 1], '.',
                         color=colors[color_name[bs]], label=labelstr)
                plt.plot(self.BSlocations[bs, 0], self.BSlocations[bs, 1], 's', color=colors[color_name[bs]])

                plt.annotate(labelstr, xy=(self.BSlocations[bs, 0], self.BSlocations[bs, 1]),
                             xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

            plt.legend()
            plt.title('Channel %d' % (c + 1))
        #plt.show()

    def _calculate_bps(self, SINR_dB):
        # mapping table of SINR => CQI => bps
        #SINR_level = [-6.9360, -5.1470, -3.1800, -1.2530, 0.7610, 2.6990, 4.6940, 6.5250, 8.5730, 10.3660,
        #              12.2890, 14.1730, 15.8880, 17.8140, 19.8290]
        SINR_level = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7,
                      14.1, 16.3, 18.7, 21.0, 22.7]
        bps_level = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
                     3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

        bps = 0
        for i in range(15):
            if (i != 14):
                if SINR_dB >= SINR_level[i] and SINR_dB < SINR_level[i+1]:
                    bps = bps_level[i]
                    break
            else:
                if SINR_dB >= SINR_level[i]:
                    bps = bps_level[i]

        return bps

    def show_SINR(self, channel_bs):

        eps1 = 10 ** (-8)
        eps2 = 10 ** (-10)

        n_channel = channel_bs.shape[0]
        for c in range(n_channel):

            receivePower = self.receivePower[channel_bs[c, :], :]
            receivePower = receivePower + eps1

            # Use the maximum received power as the served cell for each user
            max_power = np.amax(receivePower, axis=0)
            interference = np.sum(receivePower, axis=0) - max_power
            SINR = (max_power) / (interference + eps2)
            SINR_dB = 10 * np.log10(SINR)

            bps = np.zeros(self.numUEs)
            for k in range(self.numUEs):
                bps[k] = self._calculate_bps(SINR_dB[k])

            #print(np.sum(bps >= 3))

            plt.figure(figsize=(7 * self.area[0] / self.area[1], 7))

            for k in range(channel_bs.shape[1]):
                bs = channel_bs[c, k]
                labelstr = 'BS %d' % (bs + 1)

                plt.plot(self.BSlocations[bs, 0], self.BSlocations[bs, 1], 's', color='k')

                plt.annotate(labelstr, xy=(self.BSlocations[bs, 0], self.BSlocations[bs, 1]),
                             xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=10)

            selected_UE = bps < 1
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='bps < 1')
            selected_UE = (bps >= 1) & (bps < 2)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='1 < bps < 2')
            selected_UE = (bps >= 2) & (bps < 3)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='2 < bps < 3')
            selected_UE = (bps >= 3)
            plt.plot(self.UElocations[selected_UE, 0], self.UElocations[selected_UE, 1], '.', label='bps > 3')

            plt.legend()
            plt.title('Channel %d' % (c+1))
        # plt.show() # Commentend out by Ramin



    def show_range(self, area, center):

        fig = plt.figure(figsize=(9 * self.area[0] / self.area[1], 9))

        ax = fig.add_subplot(1, 1, 1)

        ax.plot(self.UElocations[:, 0], self.UElocations[:, 1], 'c.', label='UE in ray-tracing', zorder=1)

        ax.plot(self.BSlocations[:, 0], self.BSlocations[:, 1], 'rs', label='Cell')


        BSlocations_prev = self.BSlocations[0, :]
        BS_start = 1
        for n in range(1, self.numCells):
            BSlocations = self.BSlocations[n, :]
            if np.array_equal(BSlocations, BSlocations_prev) == False or n == self.numCells - 1:
                if n == self.numCells - 1:
                    BS_end = n + 1
                else:
                    BS_end = n
                labelstr = 'Cell %d - %d' % (BS_start, BS_end)
                ax.annotate(labelstr, xy=(self.BSlocations[BS_end-1, 0], self.BSlocations[BS_end-1, 1]),
                             xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=3)
                BS_start = n+1
            BSlocations_prev = BSlocations


        rect = plt.Rectangle((center[0]-area[0]/2, center[1]-area[1]/2), area[0], area[1], linewidth=2,
                             edgecolor='m', facecolor='none', zorder=4, label='simulation area')
        ax.add_patch(rect)

        ax.set_xlim([-self.area[0] / 2, self.area[0] / 2])
        ax.set_ylim([-self.area[1] / 2, self.area[1] / 2])

        ax.legend(loc='upper right')

        # plt.show() # commented out by Ramin

    '''
    def show_tilt(self, cell):
        plt.figure()

        power = self.receivePower_allTilt[:, cell, :]
        nonzero_power = power[power > 0]
        min_power = np.amin(nonzero_power)
        max_power = np.amax(nonzero_power)
        min_power_dB = 10 * np.log10(min_power)
        max_power_dB = 10 * np.log10(max_power)


        for tilt in range(0, 12):
            plt.subplot(3, 4, tilt+1)
            plt.plot(self.BSlocations[cell, 0], self.BSlocations[cell, 1], 'rs')
            labelstr = 'Cell %d' % (cell + 1)
            plt.annotate(labelstr, xy=(self.BSlocations[cell, 0], self.BSlocations[cell, 1]),
                         xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', zorder=3)
            power = self.receivePower_allTilt[tilt, cell, :]
            #nonzero_power = power[power>0]
            #min_power = np.amin(nonzero_power)
            power[power==0] = min_power
            colors = 10 * np.log10(power)
            plt.scatter(self.UElocations[:, 0], self.UElocations[:, 1], marker='.', c=colors, cmap='Blues', vmin=min_power_dB, vmax=max_power_dB)
            plt.title('Tilt %d'%(tilt+1))
            plt.colorbar()
    '''

    def CalReceivePowerAll(self, user_loc_record):
        simulation_time = user_loc_record.shape[0]
        n_user = user_loc_record.shape[1]

        self.receivePower_all = np.zeros((simulation_time, n_user, self.numCells))
        for t in range(simulation_time):
            for user in range(n_user):
                self.receivePower_all[t, user, :] = self.CalReceivePower(user_loc_record[t, user, :])

        return self.receivePower_all

    def LoadReceivePowerAll(self, receivePower_all):

        self.receivePower_all = receivePower_all


    def CalChannelGain(self, bs, location):
        # Calculate the received power based on location
        if location[0] > self.ref_loc[0]:
            grid_x_index = np.floor((location[0] - self.ref_loc[0]) / self.grid_dis).astype('int')
            x_remainder = (location[0] - self.ref_loc[0]) % self.grid_dis
            a1 = x_remainder > 0 and (grid_x_index + 1) < self.grid.shape[0]
        else:
            grid_x_index = -1

        if location[1] > self.ref_loc[1]:
            grid_y_index = np.floor((location[1] - self.ref_loc[1]) / self.grid_dis).astype('int')
            y_remainder = (location[1] - self.ref_loc[1]) % self.grid_dis
            a2 = y_remainder > 0 and (grid_y_index + 1) < self.grid.shape[1]
        else:
            grid_y_index = -1

        selected_UE = - np.ones((2, 2), dtype=int)
        if grid_x_index == -1 and grid_y_index == -1:
            # only select the upper-right grid
            selected_UE[1, 1] = self.grid[grid_x_index + 1, grid_y_index + 1]
        elif grid_x_index == -1 and grid_y_index > -1:
            # select the lower-right grid
            selected_UE[1, 0] = self.grid[grid_x_index + 1, grid_y_index]
            if a2:
                # select the upper-right grid
                selected_UE[1, 1] = self.grid[grid_x_index + 1, grid_y_index + 1]
        elif grid_x_index > -1 and grid_y_index == -1:
            # select the upper-left grid
            selected_UE[0, 1] = self.grid[grid_x_index, grid_y_index + 1]
            if a1:
                # select the upper-right grid
                selected_UE[1, 1] = self.grid[grid_x_index + 1, grid_y_index + 1]
        elif grid_x_index > -1 and grid_y_index > -1:
            selected_UE[0, 0] = self.grid[grid_x_index, grid_y_index]
            if a1:
                # select the lower-right grid
                selected_UE[1, 0] = self.grid[grid_x_index + 1, grid_y_index]
            if a2:
                # select the upper-left grid
                selected_UE[0, 1] = self.grid[grid_x_index, grid_y_index + 1]
            if a1 and a2:
                # select the upper-right grid
                selected_UE[1, 1] = self.grid[grid_x_index + 1, grid_y_index + 1]

        # determine received power
        selected_UE_list = np.where(selected_UE >= 0)[0]
        num_selected_UE = selected_UE_list.size
        if num_selected_UE == 1:
            receivePower = self.receivePower[:, selected_UE_list]
        elif num_selected_UE == 2:
            dis1 = (self.UElocations[selected_UE_list[0], 0] - location[0]) + \
                   ((self.UElocations[selected_UE_list[0], 1] - location[1]))
            dis2 = (self.UElocations[selected_UE_list[1], 0] - location[0]) + \
                   ((self.UElocations[selected_UE_list[1], 1] - location[1]))
            receivePower = dis2 / (dis1 + dis2) * self.receivePower[:, selected_UE_list[0]] \
                           + dis1 / (dis1 + dis2) * self.receivePower[:, selected_UE_list[1]]
        elif num_selected_UE == 4:
            dis_x_1 = location[0] - self.UElocations[selected_UE[0, 0], 0]
            dis_x_2 = self.UElocations[selected_UE[1, 0], 0] - location[0]
            dis_y_1 = location[1] - self.UElocations[selected_UE[0, 0], 1]
            dis_y_2 = self.UElocations[selected_UE[0, 1], 1] - location[1]

            receivePower = self.receivePower[:, selected_UE[0, 0]] * dis_x_2 * dis_y_2 + \
                           self.receivePower[:, selected_UE[1, 0]] * dis_x_1 * dis_y_2 + \
                           self.receivePower[:, selected_UE[0, 1]] * dis_x_2 * dis_y_1 + \
                           self.receivePower[:, selected_UE[1, 1]] * dis_x_1 * dis_y_1

            receivePower = receivePower / ((dis_x_1 + dis_x_2) * (dis_y_1 + dis_y_2))

        # From receivePower to ChannelGain
        transmit_power = 400  # (mW)
        eps1 = 10 ** (-8)
        receivePower = receivePower[bs] + eps1
        channelGain = receivePower / transmit_power

        return channelGain

