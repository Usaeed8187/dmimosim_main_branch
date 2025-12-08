import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

from dmimo.mimo import rankAdaptation

class linkAdaptation(Layer):
    """link adaptation for SU-MIMO and MU-MIMO"""

    def __init__(self,
                num_bs_ant,
                num_ue_ant,
                architecture,
                snrdb,
                nfft,
                N_s,
                data_sym_position,
                lookup_table_size,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.num_BS_Ant = num_bs_ant
        self.num_UE_Ant = num_ue_ant
        self.nfft = nfft
        self.architecture = architecture
        if self.architecture == 'SU-MIMO':
            snr_linear = 10**(snrdb/10)
            snr_linear = np.sum(snr_linear, axis=(2))
            self.snr_linear = np.mean(snr_linear)
            precoder= 'SVD'
        elif self.architecture == 'MU-MIMO':
            snr_linear = 10**(snrdb/10)
            self.snr_linear = np.mean(snr_linear, axis =   (0,1,3))
            precoder= 'BD'
        else:
            raise Exception(f"Rank adaptation for {self.architecture} has not been implemented.")

        self.data_sym_position = data_sym_position
        self.num_data_symbols = self.data_sym_position.shape[0]

        self.use_mmse_eesm_method = True
        self.lookup_table_size = lookup_table_size

        self.N_s = N_s
        
        self.rank_adaptation = rankAdaptation(num_bs_ant, num_ue_ant, architecture, snrdb, nfft, precoder=precoder)


    def call(self, h_est, channel_type):

        if self.architecture == "SU-MIMO":
            feedback_report  = self.generate_link_SU_MIMO(h_est, channel_type)
        elif self.architecture == "MU-MIMO":
            feedback_report = self.generate_link_MU_MIMO(h_est, channel_type)
        
        return feedback_report

    def generate_link_SU_MIMO(self, h_est, channel_type):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        total_num_symbols = h_est.shape[5]

        H_freq = tf.squeeze(h_est)
        H_freq = tf.transpose(H_freq, perm=[3,0,1,2])

        if self.use_mmse_eesm_method:

            if self.lookup_table_size == 'long':

                beta_list = np.array([1.49, 1.61, 3.36, 4.56, 6.42, 13.76, 25.16, 28.38])
                refer_sinr_db = np.array([0.2, 4.3, 5.9, 8.1, 10.3, 14.1, 18.7, 21.0])
                
                mcs_candidates = np.array([np.array([2,0.3]), np.array([2,0.6]), 
                                        np.array([4,0.37]), np.array([4,0.5]), np.array([4,0.6]), np.array([4,0.6]),
                                        np.array([6,0.55]), np.array([6,0.75]), np.array([6,0.85])])
            else:

                beta_list = np.array([1.61, 6.42, 28.38])
                refer_sinr_db = np.array([4.3, 10.3, 22.7])
                mcs_candidates = np.array([np.array([2,0.6]), np.array([4,0.66]), np.array([6,0.65])])


            qam_order_arr = np.zeros((self.N_s))
            code_rate_arr = np.zeros((self.N_s))
            cqi_snr = np.zeros((self.N_s))

            if self.N_s == 1:
                
                avg_sinr = self.snr_linear

                sinr_eff_list = []
                for beta in beta_list:
                    sinr_eff = -beta * np.log(np.mean(np.exp(-avg_sinr / beta)))
                    sinr_eff_dB = 10*np.log10(sinr_eff)
                    sinr_eff_list.append(sinr_eff_dB)
                
                curr_qam_order, curr_code_rate, cqi_snr_tmp = self.lookup_table(sinr_eff_list, refer_sinr_db, mcs_candidates)

                qam_order_arr[0] = curr_qam_order
                code_rate_arr[0] = curr_code_rate
                cqi_snr[0] = cqi_snr_tmp
                
            else:

                h_eff = self.rank_adaptation.calculate_effective_channel(self.N_s, h_est)
                n_var = self.rank_adaptation.cal_n_var(h_eff, self.snr_linear)
                mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)
                mmse_inv  = mmse_inv + n_var*tf.eye(mmse_inv.shape[-1], dtype=mmse_inv.dtype)
                mmse_inv = tf.linalg.inv(mmse_inv)
                mmse_inv = tf.matmul(h_eff, mmse_inv, adjoint_a=True)
                per_stream_sinr = self.rank_adaptation.compute_sinr(h_eff, mmse_inv, n_var)

                for stream_idx in range(self.N_s):

                    sinr_eff_list = []
                    for beta in beta_list:
                        
                        exp_term = np.exp(-per_stream_sinr[...,stream_idx] / beta)
                        if np.any(exp_term == 0):
                            sinr_eff = np.mean(per_stream_sinr)
                        else:
                            sinr_eff = -beta * np.log(np.mean(exp_term))
                        
                        sinr_eff_dB = 10*np.log10(sinr_eff)
                        sinr_eff_list.append(sinr_eff_dB)

                    curr_qam_order, curr_code_rate, cqi_snr_tmp = self.lookup_table(sinr_eff_list, refer_sinr_db, mcs_candidates)

                    qam_order_arr[stream_idx] = curr_qam_order
                    code_rate_arr[stream_idx] = curr_code_rate
                    cqi_snr[stream_idx] = cqi_snr_tmp

            return [qam_order_arr, code_rate_arr, cqi_snr]

        else:
            
            qam_order_arr = np.zeros((self.N_s, self.num_data_symbols))
            
            for sym_counter, sym_idx in enumerate(self.data_sym_position):

                u, s, vh = np.linalg.svd(H_freq[..., sym_idx])
                s_avg = np.mean(s,0)
                s_avg = s_avg[:self.N_s]

                for idx in range(self.N_s):
                    
                    tx_pow_per_stream = self.tx_pow / self.N_s
                    capacity = np.log2(1 + tx_pow_per_stream * s_avg[idx]**2 / self.noise_var_data)

                    if capacity < 4 or np.isnan(capacity):
                        curr_qam_order = 2
                    elif capacity >=4 and capacity < 6:
                        curr_qam_order = 4
                    elif capacity >=6:
                        curr_qam_order = 6
                    # elif capacity >=6 and capacity < 8:
                    #     curr_qam_order = 6
                    # elif capacity >=8:
                    #     curr_qam_order = 8
                    qam_order_arr[idx, sym_counter] = curr_qam_order
        
            qam_order_arr = np.min(qam_order_arr, -1)
        
            return qam_order_arr
    
    def generate_link_MU_MIMO(self, h_est, channel_type):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        num_rx_nodes = int((N_r - self.num_BS_Ant)/self.num_UE_Ant) + 1
        total_num_symbols = h_est.shape[5]

        h_est = h_est[0:1,...]
        H_freq = tf.squeeze(h_est)
        H_freq = tf.transpose(H_freq, perm=[3,0,1,2])

        if self.use_mmse_eesm_method:

            if self.lookup_table_size == 'long':

                beta_list = np.array([1.49, 1.61, 3.36, 4.56, 6.42, 13.76, 25.16, 28.38])
                refer_sinr_db = np.array([0.2, 4.3, 5.9, 8.1, 10.3, 14.1, 18.7, 21.0])
                
                mcs_candidates = np.array([np.array([2,0.3]), np.array([2,0.6]), 
                                        np.array([4,0.37]), np.array([4,0.5]), np.array([4,0.6]), np.array([4,0.6]),
                                        np.array([6,0.55]), np.array([6,0.75]), np.array([6,0.85])])
            else:

                beta_list = np.array([1.61, 6.42, 28.38])
                refer_sinr_db = np.array([4.3, 10.3, 22.7])
                mcs_candidates = np.array([np.array([2,0.6]), np.array([4,0.66]), np.array([6,0.65])])


            qam_order_arr = np.zeros((self.N_s, num_rx_nodes))
            code_rate_arr = np.zeros((self.N_s, num_rx_nodes))
            cqi_snr = np.zeros((self.N_s, num_rx_nodes))

            if self.N_s == 1:
                
                for rx_node_idx in range(num_rx_nodes):

                    if rx_node_idx == 0:
                        ant_indices = np.arange(self.num_BS_Ant)
                    else:
                        ant_indices = np.arange((rx_node_idx-1)*self.num_UE_Ant  + self.num_BS_Ant, rx_node_idx*self.num_UE_Ant + self.num_BS_Ant)
                    curr_sinr_linear = np.sum(self.snr_linear[ant_indices])

                    sinr_eff_list = []
                    for beta in beta_list:
                        sinr_eff = -beta * np.log(np.mean(np.exp(-curr_sinr_linear / beta)))
                        sinr_eff_dB = 10*np.log10(sinr_eff)
                        sinr_eff_list.append(sinr_eff_dB)
                    
                    curr_qam_order, curr_code_rate, cqi_snr_tmp = self.lookup_table(sinr_eff_list, refer_sinr_db, mcs_candidates)

                    qam_order_arr[0, rx_node_idx] = curr_qam_order
                    code_rate_arr[0, rx_node_idx] = curr_code_rate
                    cqi_snr[0, rx_node_idx] = cqi_snr_tmp
                
                
            else:

                h_eff = self.rank_adaptation.calculate_effective_channel(self.N_s, h_est)
                
                for rx_node_idx in range(num_rx_nodes):

                    if rx_node_idx == 0:
                        ant_indices = np.arange(self.num_BS_Ant)
                    else:
                        ant_indices = np.arange((rx_node_idx-1)*self.num_UE_Ant  + self.num_BS_Ant, rx_node_idx*self.num_UE_Ant + self.num_BS_Ant)
                    curr_sinr_linear = np.sum(self.snr_linear[ant_indices])

                    h_eff_per_node = tf.gather(h_eff, ant_indices, axis=-2)
                    
                    n_var = self.rank_adaptation.cal_n_var(h_eff_per_node, curr_sinr_linear)
                    mmse_inv = tf.matmul(h_eff_per_node, h_eff_per_node, adjoint_b=True)
                    mmse_inv  = mmse_inv + n_var*tf.eye(mmse_inv.shape[-1], dtype=mmse_inv.dtype)
                    mmse_inv = tf.linalg.inv(mmse_inv)
                    mmse_inv = tf.matmul(h_eff_per_node, mmse_inv, adjoint_a=True)
                    per_stream_sinr = self.rank_adaptation.compute_sinr(h_eff_per_node, mmse_inv, n_var)

                    for stream_idx in range(self.N_s):

                        sinr_eff_list = []
                        for beta in beta_list:
                            
                            exp_term = np.exp(-per_stream_sinr[...,stream_idx] / beta)
                            if np.mean(exp_term) == 1:
                                sinr_eff = np.mean(per_stream_sinr)
                            else:
                                sinr_eff = -beta * np.log(np.mean(exp_term))
                            
                            sinr_eff_dB = 10*np.log10(sinr_eff)
                            sinr_eff_list.append(sinr_eff_dB)

                        curr_qam_order, curr_code_rate, cqi_snr_tmp = self.lookup_table(sinr_eff_list, refer_sinr_db, mcs_candidates)

                        qam_order_arr[stream_idx, rx_node_idx] = curr_qam_order
                        code_rate_arr[stream_idx, rx_node_idx] = curr_code_rate
                        cqi_snr[stream_idx, rx_node_idx] = cqi_snr_tmp

            return [qam_order_arr, code_rate_arr, cqi_snr]
        else:
            raise Exception(f"The non-EESM methods have not been implemented.")


        
    
    def lookup_table(self, sinr_eff_list, refer_sinr_db, mcs_candidates):

        assert len(sinr_eff_list) == refer_sinr_db.shape[0]

        mcs_idx = 0
        for idx in range(refer_sinr_db.shape[0]):
            if sinr_eff_list[idx] > refer_sinr_db[idx]:
                mcs_idx += 1
        
        mcs_idx = np.max([mcs_idx-1, 0])

        [curr_qam_order, curr_code_rate] = mcs_candidates[mcs_idx, :]

        
        return curr_qam_order, curr_code_rate, refer_sinr_db[mcs_idx]
