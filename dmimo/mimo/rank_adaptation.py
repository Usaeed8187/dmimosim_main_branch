import numpy as np

import tensorflow as tf
from tensorflow.python.keras.layers import Layer

import sionna
from sionna.ofdm import ZFPrecoder, ResourceGridMapper, ResourceGrid
from sionna.utils import BinarySource
from sionna.mimo import StreamManagement
from sionna.mapping import Mapper
from sionna.utils import flatten_dims, matrix_inv

from dmimo.mimo import SVDPrecoder


class rankAdaptation(Layer):
    """Rank adaptation for SU-MIMO and MU-MIMO"""

    def __init__(self,
                num_bs_ant,
                num_ue_ant,
                architecture,
                snrdb,
                fft_size,
                precoder,
                ue_indices=None,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self.num_BS_Ant = num_bs_ant
        self.num_UE_Ant = num_ue_ant
        self.nfft = fft_size
        self.ue_indices = ue_indices

        self.architecture = architecture
        
        if self.architecture == 'SU-MIMO':
            snr_linear = 10**(snrdb/10)
            snr_linear = np.sum(snr_linear, axis=(2))
            self.snr_linear = np.mean(snr_linear)
        elif self.architecture == 'MU-MIMO':
            snr_linear = 10**(snrdb/10)
            self.snr_linear = np.mean(snr_linear, axis =   (0,1,3))
        else:
            raise Exception(f"Rank adaptation for {self.architecture} has not been implemented.")

        self.use_mmse_eesm_method = True
        self.mod = 4 # the modulation order assumed
        self.precoder = precoder
        self.A_info = 0.83
        self.B_info = 0.73

        self.threshold = 0.1

    def call(self, h_est, channel_type):

        if self.architecture == "SU-MIMO":
            feedback_report  = self.generate_rank_SU_MIMO(h_est, channel_type)
        elif self.architecture == "MU-MIMO":
            feedback_report = self.generate_rank_MU_MIMO(h_est, channel_type)

        return feedback_report

    def generate_rank_SU_MIMO(self, h_est, channel_type):
        
        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        total_num_symbols = h_est.shape[5]

        if channel_type == 'Tx_squad':
            max_rank = min(N_t, N_r) # Assumes that Tx Squad channel can always achieve max rank
        else:

            if self.use_mmse_eesm_method:

                max_rank = min(N_t, N_r)
                per_rank_rate = np.zeros((max_rank))

                for rank_idx in range(1, max_rank+1):

                    if rank_idx == 1:
                        avg_sinr = self.snr_linear
                    else:

                        h_eff = self.calculate_effective_channel(rank_idx, h_est)

                        n_var = self.cal_n_var(h_eff, self.snr_linear)

                        mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)/rank_idx + n_var
                        # mmse_inv = tf.matmul(h_eff, matrix_inv(mmse_inv), adjoint_a=True)

                        per_stream_sinr = self.compute_sinr(h_eff, mmse_inv, n_var)

                        avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                    curr_streams_rate = self.A_info * np.log2(1 + self.B_info * avg_sinr)
                    per_rank_rate[rank_idx - 1] = np.sum(curr_streams_rate)

                selected_rank = np.where(per_rank_rate == np.max(per_rank_rate))[0][0] + 1
                rate_for_selected_rank = per_rank_rate[selected_rank - 1]

                return [selected_rank, rate_for_selected_rank]


            else:
                max_rank = min(N_t, self.num_BS_Ant * 2)
                rank_capacity = np.zeros([total_num_symbols, self.nfft, max_rank])
                H_freq = tf.squeeze(h_est)
                H_freq = tf.transpose(H_freq, perm=[3,0,1,2])
                
                for sym_idx in range(total_num_symbols):
                    u, s, vh = np.linalg.svd(H_freq[..., sym_idx]) # vh: (nfft, rank, Nt)
                    
                    for rank in range(1, max_rank + 1):
                        
                        for i in range(rank):
                            
                            # tx_pow_per_stream = self.tx_pow / rank
                            # rank_capacity[sym_idx, :, rank - 1] += np.log2(1 + tx_pow_per_stream * s[:, i]**2 / self.noise_var_data)
                            snr_linear = self.snr_linear
                            snr_per_stream = snr_linear / rank
                            snr_per_stream_eff = np.min(np.mean(snr_per_stream, axis=(0,1,3)))
                            rank_capacity[sym_idx, :, rank - 1] += np.log2(1 + snr_per_stream_eff * s[:, i]**2)

                max_rank_1 = np.argmax(np.mean(rank_capacity, axis=1), axis=1) + 1
                max_rank_1 = np.min(max_rank_1)
                
                hold = 1

                ranks = np.zeros([total_num_symbols, self.nfft])
                for sym_idx in range(total_num_symbols):
                    
                    u, s, vh = np.linalg.svd(H_freq[..., sym_idx]) # vh: (nfft, rank, Nt)

                    significant_singular_values = s > self.threshold * np.max(s)
                    ranks[sym_idx, :] = np.sum(significant_singular_values, axis=1)
                
                average_rank = np.mean(ranks)
                max_rank_2 = int(np.floor(average_rank))

                hold = 1

                # max_rank = np.min([max_rank_1, max_rank_2])
                rank = max_rank_1
                
            return rank
        
    def generate_rank_MU_MIMO(self, h_est, channel_type):
        
        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        num_rx_nodes = int((N_r - self.num_BS_Ant)/self.num_UE_Ant) + 1
        total_num_symbols = h_est.shape[5]

        if channel_type == 'Tx_squad':
            max_rank = min(N_t, N_r) # Assumes that Tx Squad channel can always achieve max rank
        else:

            if self.use_mmse_eesm_method:

                max_rank = min(N_t, self.num_UE_Ant, self.num_BS_Ant)
                per_rank_rate = np.zeros((max_rank, num_rx_nodes))

                for rank_idx in range(1, max_rank+1):

                    h_eff = self.calculate_effective_channel(rank_idx, h_est)

                    for rx_node_idx in range(num_rx_nodes):
                        
                        if rx_node_idx == 0:
                            ant_indices = np.arange(self.num_BS_Ant)
                        else:
                            ant_indices = np.arange((rx_node_idx-1)*self.num_UE_Ant  + self.num_BS_Ant, rx_node_idx*self.num_UE_Ant + self.num_BS_Ant)
                        
                        h_eff_per_node = tf.gather(h_eff, ant_indices, axis=-2)

                        snr_linear = np.sum(self.snr_linear[ant_indices])
                        n_var = self.cal_n_var(h_eff_per_node, snr_linear)

                        mmse_inv = tf.matmul(h_eff_per_node, h_eff_per_node, adjoint_b=True)/rank_idx + n_var
                        # mmse_inv = tf.linalg.inv(mmse_inv)

                        per_stream_sinr = self.compute_sinr(h_eff_per_node, mmse_inv, n_var)
                        if rank_idx == 1:
                            per_stream_sinr = tf.gather(per_stream_sinr, rx_node_idx, axis=-1)
                        elif rank_idx == 2:
                            per_stream_sinr = per_stream_sinr

                        avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                        curr_streams_rate = self.A_info * np.log2(1 + self.B_info * avg_sinr)

                        per_rank_rate[rank_idx - 1, rx_node_idx] = np.sum(curr_streams_rate)
                
                # TODO: Add per-user rank selection
                min_per_rank_rate = np.min(per_rank_rate, axis=1)
                
                selected_rank = np.where(min_per_rank_rate == np.max(min_per_rank_rate))[0][0] + 1
                rate_for_selected_rank = min_per_rank_rate[selected_rank - 1]

                return [selected_rank, rate_for_selected_rank]
            
            else:

                max_rank = min(N_t, self.num_UE_Ant)
                rank = max_rank
                rank_capacity = np.zeros([total_num_symbols, self.nfft, max_rank])
                H_freq = tf.squeeze(h_est)
                H_freq = tf.transpose(H_freq, perm=[3,0,1,2])

                num_UEs = int((N_r - self.num_BS_Ant) / self.num_UE_Ant)
                num_rx_nodes = num_UEs + 1

                for rx_node_id in range(num_rx_nodes):

                    if rx_node_id == 0:
                        H_freq_temp = H_freq[:,:self.num_BS_Ant, ...]
                        ant_idx = tf.range(0, self.num_BS_Ant)
                    else:
                        ant_idx = tf.range(self.num_BS_Ant + (rx_node_id-1)*self.num_UE_Ant, self.num_BS_Ant + rx_node_id * self.num_UE_Ant)
                        H_freq_temp = tf.gather(H_freq, ant_idx, axis=1)
                    
                    snr_linear_nodewise = self.snr_linear[:,:,ant_idx,:]

                    for sym_idx in range(total_num_symbols):
                        
                        u, s, vh = np.linalg.svd(H_freq_temp[..., sym_idx]) # vh: (nfft, rank, Nt)
                        
                        for rank in range(1, max_rank + 1):
                            
                            for i in range(rank):
                                snr_per_stream = snr_linear_nodewise / rank
                                snr_per_stream_eff = np.min(np.mean(snr_per_stream, axis=(0,1,3)))
                                rank_capacity[sym_idx, :, rank - 1] += np.log2(1 + snr_per_stream_eff * s[:, i]**2)               
                    max_rank_temp = np.argmax(np.mean(rank_capacity, axis=1), axis=1) + 1
                    if np.min(max_rank_temp) < rank:
                        rank = np.min(max_rank_temp)

                return rank

    
    def calculate_effective_channel(self, stream_idx, h_est):
        
        if self.precoder == 'SVD':
            v, u_h = self.generate_svd_precoding(stream_idx, h_est) # calculating the svd precoder
            # Select the columns of v according to number of spatial streams
            u_h = tf.gather(u_h, np.arange(stream_idx), axis=4)
        elif self.precoder == 'BD':
            v, u_h = self.generate_bd_precoding(stream_idx, h_est) # calculating the svd precoder

        h_est_reshaped = tf.transpose(h_est, [0, 1, 3, 5, 6, 2, 4])
        h_est_reshaped = tf.cast(h_est_reshaped, dtype=v.dtype)

        h_eff = tf.matmul(h_est_reshaped, v)

        if self.precoder == 'SVD':
            h_eff = tf.matmul(u_h, h_eff)
        elif self.precoder == 'BD':
            h_eff = h_eff

        return h_eff



    def generate_svd_precoding(self, num_streams, h):

        num_tx_ant = h.shape[-3]
        num_streams = num_streams
        assert num_streams <= num_tx_ant, "Number of stream should not exceed number of antennas"

        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Transformations to bring h in the desired shapes

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        # dimensions:
        # h_pc_desired: [batch_size, num_tx, num_ofdm_sym, fft_size, num_streams_per_tx, num_tx_ant]
        # Compute SVD of channel matrix for precoding
        s, u, v = tf.linalg.svd(h_pc_desired, compute_uv=True)

        # Select the columns of v according to num_streams
        v = tf.gather(v, np.arange(num_streams), axis=5)

        # Make the signs of eigen vectors consistent
        v = tf.sign(v[..., :1, :]) * v

        u_h = tf.transpose(u, [0,1,2,3,5,4], conjugate=True)

        return v, u_h
    
    def generate_bd_precoding(self, num_streams, h):

        num_tx_ant = h.shape[-3]
        num_streams = num_streams
        assert num_streams <= num_tx_ant, "Number of stream should not exceed number of antennas"

        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]

        # Transformations to bring h in the desired shapes

        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        num_user = int((h_pc_desired.shape[-2] - self.num_BS_Ant)  / self.num_UE_Ant) + 1
        # num_user = num_user 

        v_all = []
        u_all = []
        for k in range(num_user):
            # Step 1: block diagonalization to minimize MUI
            if k == 0:
                ue_indices = np.arange(self.num_BS_Ant)
                ue_indices_comp = [np.arange(h_pc_desired.shape[-2])[i] for i in range(h_pc_desired.shape[-2]) if i not in ue_indices]
                num_rx_ant = ue_indices.shape[0]
            else:
                ue_indices = np.arange((k-1)*self.num_UE_Ant  + self.num_BS_Ant, k*self.num_UE_Ant + self.num_BS_Ant)
                ue_indices_comp = [np.arange(h_pc_desired.shape[-2])[i] for i in range(h_pc_desired.shape[-2]) if i not in ue_indices]
                num_rx_ant = ue_indices.shape[0]
            H_t = tf.gather(h_pc_desired, indices=ue_indices_comp, axis=-2)  # [..., total_rx_ant-num_rx_ant, num_tx_ant]
            s, u, v = tf.linalg.svd(H_t, compute_uv=True, full_matrices=True)
            # Make the signs of eigen vectors consistent
            v = tf.sign(v[..., :1, :]) * v
            # null space bases for use k
            v_c = v[..., -num_rx_ant:]  # [..., num_tx_ant, num_rx_ant]
            # effective channel for user k
            H_k = tf.gather(h_pc_desired, indices=ue_indices, axis=-2)  # [..., num_rx_ant, num_tx_ant]
            H_eff = tf.linalg.matmul(H_k, v_c)  # [..., num_rx_ant, num_rx_ant]

            # Step 2: compute SVD for individual user
            s2, u2, v2 = tf.linalg.svd(H_eff, compute_uv=True, full_matrices=True)
            
            w = tf.linalg.adjoint(tf.sign(v2[..., :1, :]) * u2)
            w = w[..., :num_streams, :]
            u_all.append(w)
            
            # Make the signs of eigen vectors consistent
            v2 = tf.sign(v2[..., :1, :]) * v2
            # rank adaptation
            v2 = v2[..., :num_streams]
            ss = tf.linalg.diag(tf.cast(1.0 / s2[..., :num_streams], tf.complex64))
            v2 = tf.linalg.matmul(v2, ss)
            v_eff = tf.linalg.matmul(v_c, v2)  # [..., num_tx_ant, num_rx_ant]
            v_all.append(v_eff)

        # combine v_eff for all users
        v_bd = tf.concat(v_all, axis=-1)  # [..., num_tx_ant, num_streams_per_tx]
        u_bd = tf.concat(u_all, axis=-1)  # [..., num_tx_ant, num_streams_per_tx]
        return v_bd, u_bd

    def compute_sinr(self, h_eff, mmse_inv, n_var):
        N_s = h_eff.shape[-1]
        sinr_list = []

        for i in range(N_s):
            h_i = tf.gather(h_eff, i, axis=6)
            h_i = tf.expand_dims(h_i, -1)
            
            # Compute the numerator: |diag(h_i^H * MMSE_R_inv * h_i)|^2
            numerator = tf.abs(tf.linalg.diag_part(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, h_i))))**2
            
            # Compute the denominator: n_var * diag(real(h_i^H * MMSE_R_inv * MMSE_R_inv^H * h_i)) + sum(|diag(h_i^H * MMSE_R_inv * h_j)|^2)
            mmse_inv_h_i = tf.matmul(tf.linalg.adjoint(mmse_inv), h_i)
            real_part = tf.linalg.diag_part(tf.math.real(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, mmse_inv_h_i))))
            interference_sum = tf.zeros_like(real_part)

            for j in range(N_s):
                if j != i:
                    h_j = tf.gather(h_eff, j, axis=6)
                    h_j = tf.expand_dims(h_j, -1)
                    interference_sum += tf.abs(tf.linalg.diag_part(tf.matmul(tf.linalg.adjoint(h_i), tf.matmul(mmse_inv, h_j))))**2
            
            denominator = n_var * real_part + interference_sum
            
            # Calculate SINR for h_i
            sinr_i = numerator / denominator
            sinr_list.append(sinr_i)

        # Stack the SINR values to form the final SINR tensor
        sinr = tf.stack(sinr_list, axis=-1)
        return sinr

    def eesm_average(self, sinr, er, mod):

        if mod == 2:  # QPSK
            beta = 0.413 * er + 1.3661
        elif mod == 4:  # 16-QAM
            beta = 4.4492 * er**2 + 4.5655 * er + 1.2982
        elif mod == 6:  # 64-QAM
            beta = 4.1182 * np.exp(2.4129 * er)
        else:
            raise ValueError('Supported modulation sizes are 2, 4, and 6 only.')

        N = int(np.size(sinr) / sinr.shape[-1])
        exp_sum = np.sum(np.exp(-sinr / beta), axis=(0,1,2,3,4,5))
        exp_sum = 1 / N * exp_sum

        if np.any(exp_sum == 0):
            eesm_avg_sinr = np.mean(sinr)
        else:
            eesm_avg_sinr = -beta * np.log(exp_sum)

        return eesm_avg_sinr

    def cal_n_var(self, h_eff, snr_linear):
        
        prod = tf.matmul(h_eff, h_eff, adjoint_b=True)
        sig_pow = np.abs(np.mean(np.trace(prod, axis1=-2, axis2=-1)))

        n_var = sig_pow / snr_linear

        return n_var
