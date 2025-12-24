import copy
import numpy as np
import tensorflow as tf
import os

from dmimo.config import Ns3Config, RCConfig
from dmimo.channel import lmmse_channel_estimation

class standard_rc_pred_freq_mimo:

    def __init__(self, architecture, num_rx_ant=8, ns3cfg=None):
        
        if ns3cfg is None:
            ns3_config = Ns3Config()
        else:
            ns3_config = ns3cfg
        self.ns3_config = ns3_config
        rc_config = RCConfig()

        self.nfft = 512  # TODO: remove hardcoded param value
        self.num_rx_ant = num_rx_ant  # TODO: use node selection mask
        
        if architecture == 'baseline':
            self.N_t = ns3_config.num_bs_ant
            self.N_r = ns3_config.num_bs_ant
        elif architecture == 'SU_MIMO':
            self.N_t = ns3_config.num_bs_ant + ns3_config.num_ue_ant * ns3_config.num_txue
            self.N_r = ns3_config.num_bs_ant * 2
        elif architecture == 'MU_MIMO':
            self.N_t = ns3_config.num_bs_ant + ns3_config.num_ue_ant * ns3_config.num_txue
            self.N_r = num_rx_ant
        else:
            raise ValueError("\n The architecture specified is not correct")

        if rc_config.treatment == 'SISO':
            self.N_t = 1
            self.N_r = 1

        self.N_n = rc_config.num_neurons
        self.sparsity = rc_config.W_tran_sparsity
        self.spectral_radius = rc_config.W_tran_radius
        self.max_forget_length = rc_config.max_forget_length
        self.initial_forget_length = rc_config.initial_forget_length
        self.forget_length = rc_config.initial_forget_length
        self.forget_length_search_step = rc_config.forget_length_search_step
        self.input_scale = rc_config.input_scale
        self.window_length = rc_config.window_length
        self.learning_delay = rc_config.learning_delay
        self.reg = rc_config.regularization
        self.enable_window = rc_config.enable_window
        self.history_len = rc_config.history_len

        seed = 10
        self.RS = np.random.RandomState(seed)
        self.type = rc_config.type # 'real', 'complex'
        # self.type = 'real' # 'real', 'complex'
        # if self.type == 'real':
        #     self.N_in = self.N_r * self.window_length * 2
        #     self.N_out = self.N_t * 2
        #     self.S_0 = np.zeros([self.N_n])
        # else:
        #     self.N_in = self.N_r * self.window_length
        #     self.N_out = self.N_t
        #     self.S_0 = np.zeros([self.N_n], dtype='complex')
        if self.enable_window:
            self.N_in = self.nfft * self.N_r * self.N_t * self.window_length
        else:
            self.N_in = self.nfft * self.N_r * self.N_t
        self.N_out = self.nfft * self.N_r * self.N_t
        self.S_0 = np.zeros([self.N_n], dtype='complex')

        self.init_weights()
        self.W_out = self.RS.randn(self.N_out, self.N_n + self.N_in)

        # self.digit_mod = Dataset.digit_mod
        # self.ofdm_mod = Dataset.ofdm_mod
        # self.ofdm_structure = Dataset.ofdm_structure
        # self.noise_var_decode = Dataset.noise_var_decode
        # if self.ofdm_structure == 'WiFi_OFDM':
        #     self.train_rls = RC_para['train_rls']
        # else:
        self.train_rls = False
        self.DF_rls = rc_config.DF_rls

        if self.train_rls or self.DF_rls:
            # for RLS algorithm
            self.psi = np.identity(self.N_in + self.N_n)
            if self.type == 'complex':
                self.psi = self.psi.astype(complex)
            self.psi_inv = np.linalg.inv(self.psi)

            # self.RLS_lambda = 0.9995, 0.9998
            # self.RLS_w = 1 / (1 + np.exp(-(self.EbNo - 11)))

            # self.RLS_lambda = 0.99999
            self.RLS_lambda = rc_config.RLS_lambda
            self.RLS_w = 1

        # Buffers for incremental CSI history updates
        self.csi_history_buffer = None
        self.csi_history_slots = None

    def reset_csi_history(self):
        """Reset cached CSI history."""

        self.csi_history_buffer = None
        self.csi_history_slots = None

    def _load_or_estimate_channel(self, slot_idx, rg_csi, dmimo_chans, cfo_vals, sto_vals, estimated_channels_dir):
        """Load a single channel estimate from disk or run LMMSE estimation."""

        folder_path = estimated_channels_dir + "_rx_{}_tx_{}".format(
            self.ns3_config.num_bs_ant + self.ns3_config.num_ue_ant * self.ns3_config.num_rxue_sel,
            self.ns3_config.num_bs_ant + self.ns3_config.num_ue_ant * self.ns3_config.num_txue_sel,
        )
        file_path = "{}/dmimochans_{}".format(folder_path, slot_idx)

        try:
            data = np.load("{}.npz".format(file_path))
            h_freq_csi = data['h_freq_csi']
        except Exception:
            # h_freq_csi has shape [batch_size, num_rx, num_rx_ant, num_tx, num_txs_ant, num_ofdm_sym, fft_size]
            h_freq_csi, _ = lmmse_channel_estimation(
                dmimo_chans,
                rg_csi,
                slot_idx=slot_idx,
                cfo_vals=cfo_vals,
                sto_vals=sto_vals,
            )
            os.makedirs(folder_path, exist_ok=True)
            np.savez(file_path, h_freq_csi=h_freq_csi)

        return np.expand_dims(h_freq_csi, axis=0)

            


    def get_csi_history(self, first_slot_idx, csi_delay, rg_csi, dmimo_chans, cfo_vals=[0], sto_vals=[0], estimated_channels_dir=None):

        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)
        channel_history_slots = np.arange(first_csi_history_idx, first_slot_idx, csi_delay)

        # Initialize or rebuild buffer when unavailable or mismatched
        if (
            self.csi_history_buffer is None
            or self.csi_history_slots is None
            or len(self.csi_history_slots) != len(channel_history_slots)
        ):
            h_freq_csi_list = [
                self._load_or_estimate_channel(
                    slot_idx, rg_csi, dmimo_chans, cfo_vals, sto_vals, estimated_channels_dir
                )
                for slot_idx in channel_history_slots
            ]
            self.csi_history_buffer = np.concatenate(h_freq_csi_list, axis=0)
            self.csi_history_slots = channel_history_slots
            return self.csi_history_buffer

        # If slots are unchanged, return cached buffer
        if np.array_equal(self.csi_history_slots, channel_history_slots):
            return self.csi_history_buffer

        # If slots advanced by one step, update buffer incrementally
        if np.array_equal(self.csi_history_slots[1:], channel_history_slots[:-1]):
            newest_slot_idx = channel_history_slots[-1]
            new_entry = self._load_or_estimate_channel(
                newest_slot_idx, rg_csi, dmimo_chans, cfo_vals, sto_vals, estimated_channels_dir
            )
            self.csi_history_buffer = np.concatenate([self.csi_history_buffer[1:], new_entry], axis=0)
            self.csi_history_slots = channel_history_slots
            return self.csi_history_buffer

        # Fallback: rebuild buffer if slot progression is unexpected
        h_freq_csi_list = [
            self._load_or_estimate_channel(
                slot_idx, rg_csi, dmimo_chans, cfo_vals, sto_vals, estimated_channels_dir
            )
            for slot_idx in channel_history_slots
        ]
        self.csi_history_buffer = np.concatenate(h_freq_csi_list, axis=0)
        self.csi_history_slots = channel_history_slots
        
        return self.csi_history_buffer

    def get_ideal_csi_history(self, first_slot_idx, csi_delay, dmimo_chans, batch_size=1):
        
        # Get perfect channel estimate history starting from (csi_delay * self.history_len) slots in the past to the most up-to-date fed back estimate
        # Here "first_slot_idx" is used as the index of the current slot. 
        # e.g. if first_slot_idx is 12, we are in the 12th slot. 
        # if csi_delay = 6 and self.history_len = 2, then we want to use the estimates for slots 12-6x2 = 0 and 12-6x1 = 6 for training
        # and we try to predict the unknown channel for slot 12, which is the index of the current slot

        first_csi_history_idx = first_slot_idx - (csi_delay * self.history_len)  # TODO: currently only for self.history_len = 2
        channel_history_slots = np.arange(first_csi_history_idx, first_slot_idx, csi_delay)
            
        h_freq_csi_list = []
        for loop_idx, slot_idx in enumerate(channel_history_slots):
            h_freq_csi , _, _ = dmimo_chans.load_channel(slot_idx=slot_idx, batch_size=batch_size)
            h_freq_csi_list.append(np.expand_dims(h_freq_csi, axis=0))

        h_freq_csi_history = np.concatenate(h_freq_csi_list, axis=0)

        return h_freq_csi_history


    def rc_vectorized_predict(self, h_freq_csi_history):

        h_freq_csi_history = np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6])

        batch_size = h_freq_csi_history.shape[1]  # TODO: Not currently used
        num_training_slots =  h_freq_csi_history.shape[0]

        channel_train_input = h_freq_csi_history[0, ...]  # TODO: Batch-wise treatment
        channel_train_gt = h_freq_csi_history[-1, ...]  # TODO: Batch-wise treatment

        num_batches = h_freq_csi_history.shape[1]
        num_rx_nodes = h_freq_csi_history.shape[2]
        num_tx_nodes = h_freq_csi_history.shape[4]

        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=complex)
        for batch_idx in range(num_batches):
            for rx_node in range(num_rx_nodes):
                for tx_node in range(num_tx_nodes):
                    
                    channel_train_input_temp = channel_train_input[batch_idx, rx_node, :, tx_node, ...]
                    channel_train_input_temp = channel_train_input_temp.reshape(-1, channel_train_input_temp.shape[-1])
                    channel_train_gt_temp = channel_train_gt[batch_idx, rx_node, :, tx_node, ...]
                    channel_train_gt_temp = channel_train_gt_temp.reshape(-1, channel_train_gt_temp.shape[-1])

                    self.fitting_time(channel_train_input_temp, channel_train_gt_temp)

                    channel_test_input = channel_train_gt_temp
                    channel_pred_temp = self.test_train_predict(channel_test_input)
                    channel_pred_temp = channel_pred_temp.reshape(channel_train_input[batch_idx, rx_node, :, tx_node, ...].shape)
                    chan_pred[batch_idx, rx_node, :, tx_node, ...] = channel_pred_temp

        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        chan_pred = tf.convert_to_tensor(chan_pred)
        return chan_pred
    
    def rc_siso_predict(self, h_freq_csi_history):
        
        if tf.rank(h_freq_csi_history).numpy() == 8:
            h_freq_csi_history = np.asarray(h_freq_csi_history).transpose([0,1,2,3,4,5,7,6])
            num_batches = h_freq_csi_history.shape[1]
            num_rx_nodes = h_freq_csi_history.shape[2]
            num_tx_nodes = h_freq_csi_history.shape[4]
            num_tx_antennas = h_freq_csi_history.shape[5]
            num_rx_antennas = h_freq_csi_history.shape[3]
        else:
            raise ValueError("\n The dimensions of h_freq_csi_history are not correct")

        batch_size = h_freq_csi_history.shape[1]  # TODO: Not currently used
        num_training_slots = h_freq_csi_history.shape[0]

        channel_train_input = h_freq_csi_history[:-1, ...]
        channel_train_gt = h_freq_csi_history[1:, ...]
        # channel_train_input = h_freq_csi_history[:num_training_slots//2, ...]  # TODO: Batch-wise treatment
        # channel_train_gt = h_freq_csi_history[num_training_slots//2:, ...]  # TODO: Batch-wise treatment
        # channel_train_input = channel_train_input.transpose([1, 2, 3, 4, 5, 6, 0, 7])
        # channel_train_input = channel_train_input.reshape(channel_train_input.shape[:6] + (-1,))

        # chan_pred_train = np.zeros(h_freq_csi_history[:2,...].shape, dtype=complex)
        chan_pred = np.zeros(h_freq_csi_history[0,...].shape, dtype=complex)
        for batch_idx in range(num_batches):
            for rx_node in range(num_rx_nodes):
                for tx_node in range(num_tx_nodes):
                    for tx_ant in range(num_tx_antennas):
                        for rx_ant in range(num_rx_antennas):
                    
                            channel_train_input_temp = channel_train_input[:, batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...]
                            channel_train_input_temp = channel_train_input_temp.transpose([1, 0, 2]).reshape(self.nfft, -1)
                            # channel_train_input_temp = channel_train_input_temp.reshape(-1, channel_train_input_temp.shape[-1])
                            channel_train_gt_temp = channel_train_gt[:, batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...]
                            channel_train_gt_temp = channel_train_gt_temp.transpose([1, 0, 2]).reshape(self.nfft, -1)
                            # channel_train_gt_temp = channel_train_gt_temp.reshape(-1, channel_train_gt_temp.shape[-1])

                            curr_train = self.fitting_time(channel_train_input_temp, channel_train_gt_temp)
                            # curr_train = curr_train.reshape(channel_train_input[batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...].shape)
                            # chan_pred_train[:, batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...] = curr_train

                            # channel_test_input = channel_train_gt_temp
                            channel_test_input = channel_train_gt[-1, batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...]
                            channel_pred_temp = self.test_train_predict(channel_test_input)
                            channel_pred_temp = channel_pred_temp.reshape(channel_train_input[0, batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...].shape)
                            chan_pred[batch_idx, rx_node, rx_ant, tx_node, tx_ant, ...] = channel_pred_temp

        chan_pred = chan_pred.transpose([0,1,2,3,4,6,5])
        chan_pred = tf.convert_to_tensor(chan_pred)
        # train_nmse = self.cal_nmse(channel_train_gt, chan_pred_train)
        # print(f"train nmse: {train_nmse}")
        return chan_pred


    def init_weights(self):
        if self.type == 'real':
            self.W = self.sparse_mat(self.N_n)
            self.W_in = 2 * (self.RS.rand(self.N_n, self.N_in) - 0.5)
            self.W_tran = np.concatenate([self.W, self.W_in], axis=1)
        else:
            self.W = self.sparse_mat(self.N_n)
            self.W_in = 2 * (self.RS.rand(self.N_n, self.N_in) - 0.5)
            self.W_tran = np.concatenate([self.W, self.W_in], axis=1)

    def sparse_mat(self, m):
        if self.type == 'real':
            W = self.RS.rand(m, m) - 0.5
            W[self.RS.rand(*W.shape) < self.sparsity] = 0
        else:
            W = 2*(self.RS.rand(m, m) - 0.5) + 2j*(self.RS.rand(m, m) - 0.5)
            W[self.RS.rand(*W.shape) < self.sparsity] = 0+1j*0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (self.spectral_radius / radius)
        return W

    def complex_to_real_target(self, Y_target_2D):
        Y_target_2D_real_list = []
        for t in range(self.N_t):
            target = Y_target_2D[t, :].reshape(1, -1) # (1, N_symbols * (N_fft+N_cp))
            real_target = np.concatenate((np.real(target), np.imag(target)), axis=0)  # (2, N_symbols * (N_fft+N_cp))
            Y_target_2D_real_list.append(real_target)
        Y_target_2D_real = np.concatenate(Y_target_2D_real_list, axis=0)
        return Y_target_2D_real

    def fitting_time(self, channel_input, channel_output):
        # @todo only work for SISO
        # channel_input: [32, ]
        Y_2D = channel_input
        Y_target_2D = channel_output

        if self.type == 'real':
            Y_target_2D = self.complex_to_real_target(Y_target_2D)

        obj_value_delay = []
        W_out_delay = []
        delay_value = []

        # if self.learning_delay:
        for d in np.arange(self.initial_forget_length, self.max_forget_length, self.forget_length_search_step):
            self.forget_length = d # delay
            if self.enable_window:
                Y_2D_new = self.form_window_input_signal(Y_2D)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
            else:
                Y_2D_new = np.concatenate([Y_2D, np.zeros([Y_2D.shape[0], self.forget_length], dtype=Y_2D.dtype)], axis=1)
            S_2D_transit = self.state_transit(Y_2D_new * self.input_scale) # (16, 640) (N_n, N_symbols * (N_fft + N_cp))
            Y_2D_new = Y_2D_new[:, self.forget_length:] # (32, 640) (N_r * window_length, N_symbols * (N_fft + N_cp))
            S_2D = np.concatenate([S_2D_transit, Y_2D_new], axis=0) # (N_n + N_r * window_length, N_symbols * (N_fft + N_cp))
            assert Y_target_2D.shape[0] == self.N_out
            if self.reg == 0:
                self.W_out = Y_target_2D @ np.linalg.pinv(S_2D)
            else:
                self.W_out = Y_target_2D @ self.reg_p_inv(S_2D)
            pred_channel = self.W_out @ S_2D
            obj_value_delay.append(self.cal_nmse(Y_target_2D, pred_channel))
            W_out_delay.append(self.W_out)
            delay_value.append(d)
            # print('Delay:', d, ', INV: The fitting NMSE is', obj_value_delay[-1])

            if self.DF_rls:
                for i in range(S_2D.shape[1]):
                    self.psi_inv = self.RLS_psi_inv(S_2D[:, i], self.psi_inv)

        indx = np.argmin(obj_value_delay)
        self.forget_length = delay_value[indx]
        self.W_out = W_out_delay[indx]
        # print(f'Optimal delay is {self.forget_length}, min NMSE is {obj_value_delay[indx]}')
        return pred_channel

    def cal_nmse(self, H, H_hat):
        mse = np.sum(np.abs(H - H_hat) ** 2)
        normalization_factor = np.sum((np.abs(H) + np.abs(H_hat)) ** 2)
        nmse = mse / normalization_factor
        return nmse

    def reg_p_inv(self, X):
        N = X.shape[0]
        return np.conj(X.T)@np.linalg.pinv(X@np.conj(X.T)+self.reg*np.eye(N))

    def RLS_psi_inv(self, extended_state, psi_inv_pre):
        lambda_temp = self.RLS_lambda
        # lambda_temp=1
        extended_state = extended_state.reshape((-1, 1))
        u = np.matmul(psi_inv_pre, extended_state)  # (N_reservoir + N_in, 1)
        k = float(1)/(lambda_temp + np.matmul(np.conj(extended_state.T), u)) * u  # (N_reservoir + N_in, 1)
        psi_inv_current = float(1) /lambda_temp * (psi_inv_pre - np.matmul(k, np.matmul(np.conj(extended_state.T), psi_inv_pre)))
        return psi_inv_current

    def form_window_input_signal(self, Y_2D_complex):
        # Y_2D: [N_r, N_symbols * (N_fft + N_cp)]
        if self.type == 'real':
            Y_2D = np.concatenate((Y_2D_complex.real, Y_2D_complex.imag), axis=0)
        else:
            Y_2D = copy.deepcopy(Y_2D_complex)
        Y_2D = np.concatenate([Y_2D, np.zeros([Y_2D.shape[0], self.forget_length], dtype=Y_2D.dtype)], axis=1) # [N_r, N_symbols * (N_fft + N_cp) + delay]
        Y_2D_window = []

        for n in range(self.window_length):
            shift_y_2d = np.roll(Y_2D, shift=n, axis=-1)
            if self.type == 'real':
                shift_y_2d[:, :n] = 0.
            else:
                shift_y_2d[:, :n] = 0. + 0.j
            Y_2D_window.append(shift_y_2d) # a method to explore

        # Y_2D_window = np.concatenate(Y_2D_window, axis = 0) # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        if self.type == 'real':
            Y_2D_window = np.concatenate(Y_2D_window, axis = 1).reshape(self.N_in, -1)
        else:
            Y_2D_window = np.concatenate(Y_2D_window, axis=1).reshape(self.N_in, -1)
        return Y_2D_window

    def test_train_predict(self, channel_train_input):
        self.S_0 = np.zeros([self.N_n], dtype='complex')
        Y_2D_org = channel_train_input
        if self.enable_window:
            Y_2D = self.form_window_input_signal(Y_2D_org)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        else:
            Y_2D = np.concatenate([Y_2D_org, np.zeros([Y_2D_org.shape[0], self.forget_length], dtype=Y_2D_org.dtype)], axis=1)
        S_2D = self.state_transit(Y_2D * self.input_scale)
        Y_2D = Y_2D[:, self.forget_length:]
        S_2D = np.concatenate([S_2D, Y_2D], axis=0)
        curr_channel_pred = self.W_out @ S_2D
        return curr_channel_pred

    def test_pred(self, channel_test_input, pred_num, test_train_error=False):
        # Y_2D_org = Y_3D.reshape([self.N_r, -1])
        # channel_test_input: [num_taps, 1]
        # @todo only work for SISO
        if test_train_error:
            Y_2D_org = channel_test_input[:, 0].reshape(-1, 1).copy()
        else:
            Y_2D_org = channel_test_input.reshape(-1, 1).copy()
        num_taps = channel_test_input.shape[0]
        channel_pred = np.zeros([num_taps, pred_num], dtype=complex)

        for i in range(pred_num):
            if self.enable_window:
                Y_2D = self.form_window_input_signal(Y_2D_org)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
            else:
                Y_2D = np.concatenate(
                    [Y_2D_org, np.zeros([Y_2D_org.shape[0], self.forget_length], dtype=Y_2D_org.dtype)], axis=1)
            S_2D = self.state_transit(Y_2D * self.input_scale)
            Y_2D = Y_2D[:, self.forget_length:]
            S_2D = np.concatenate([S_2D, Y_2D], axis=0)
            curr_channel_pred = self.W_out @ S_2D
            Y_2D_org = curr_channel_pred.copy()
            channel_pred[:, i] = curr_channel_pred.reshape(-1)

        return channel_pred

    def real_to_complex_predict(self, Tx_data_time_symbols_2D):
        predict_complex_list = []
        for t in range(self.N_t):
            curr_complex = Tx_data_time_symbols_2D[t * 2] + 1j * Tx_data_time_symbols_2D[t * 2 + 1]
            predict_complex_list.append(curr_complex.reshape(1, -1))
        predict_complex = np.concatenate(predict_complex_list, axis=0)
        return predict_complex

    def forward(self, Y_2D_complex):
        # add the length of zeros as forget_length, Y_2D: [N_r, N_symbols * (N_fft+N_cp)]
        Y_2D = self.form_window_input_signal(Y_2D_complex)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        S_2D = self.state_transit(Y_2D*self.input_scale)
        return S_2D

    def state_transit(self, Y_2D):
        # add the length of zeros as forget_length, Y_2D: [N_r, N_symbols * (N_fft+N_cp)]
        # Y_2D = self.form_window_signal_extended_state(
        #     Y_2D)  # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]
        # Y_2D = self.form_window_input_signal(Y_2D_complex) # [N_r * window_length, N_symbols * (N_fft + N_cp)+delay]

        T = Y_2D.shape[-1] # number of samples
        # self.S_0 = np.zeros([self.N_n], dtype='complex')
        S_1D = copy.deepcopy(self.S_0)
        S_2D = []
        for t in range(T):
            if self.type == 'real':
                S_1D = np.tanh(self.W_tran @ np.concatenate([S_1D, Y_2D[:, t]], axis=0)) + 1e-6 * (self.RS.rand(self.N_n) - 0.5)
            else:
                S_1D = self.complex_tanh(self.W_tran@np.concatenate([S_1D, Y_2D[:, t]], axis = 0))
            S_2D.append(S_1D)

        S_2D = np.stack(S_2D, axis=1)
        # self.S_0 = S_1D
        return S_2D[:, self.forget_length:]
        # return S_2D

    # def state_transit_RLS(self, Y_2D_windowed):
    #     T = Y_2D_windowed.shape[-1]  # number of samples
    #     S_1D = copy.deepcopy(self.S_0)
    #     S_2D = []
    #     for t in range(T):
    #         if self.type == 'real':
    #             S_1D = np.tanh(self.W_tran @ np.concatenate([S_1D, Y_2D_windowed[:, t]], axis=0)) + 1e-6 * (
    #                         self.RS.rand(self.N_n) - 0.5)
    #         else:
    #             S_1D = self.complex_tanh(self.W_tran @ np.concatenate([S_1D, Y_2D_windowed[:, t]],
    #                                                               axis=0)) + 1e-6 * (self.RS.rand(self.N_n) - 0.5) # ? line 148 of pyESN.py
    #         S_2D.append(S_1D)
    #
    #     S_2D = np.stack(S_2D, axis=1)
    #     self.S_0 = S_1D
    #     return S_2D

    def complex_tanh(self, Y):
        return np.tanh(np.real(Y)) + 1j * np.tanh(np.imag(Y))
