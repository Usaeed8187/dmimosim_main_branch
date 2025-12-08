import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

class quantized_CSI_feedback(Layer):
    """CSI feedback report generation"""

    def __init__(self,
                method,
                codebook_selection_method,
                num_tx_streams,
                architecture,
                snrdb,
                wideband=False,
                total_bits=None,
                VectorLength=None,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.nfft = 512
        self.subcarriers_per_RB = 12

        self.codebook_selection_method = codebook_selection_method # 'rate', 'chordal_dist'

        self.method = method
        if self.method == '5G':
            self.N_1 = 2 # Number of quantization points in the horizontal dimension
            self.N_2 = 1 # Number of quantization points in the vertical dimension
            self.O_1 = 4 # Horizontal oversampling factor
            self.O_2 = 1 # Vertical oversampling factor
        elif self.method == 'RVQ':
            #Initialization for the RVQ quantization
            # This can be any even division of these but in our case it should always be the full number of bits. 
            self.codebook_size=int(2**total_bits)
            self.bits_per_codeword=total_bits
            self.num_codewords = total_bits // self.bits_per_codeword
            
            #VectorLength=data.shape[3]*2 generally its the number of TX times 2 RX antennas since all nodes have 2 RX antennas. 
            # Generate random codebook (complex numbers), you can use a uniform distribution but I have seen reference that guassian is better for channels. 
            #self.codebook = np.random.randn(self.codebook_size,VectorLength) + 1j * np.random.randn(self.codebook_size,VectorLength)
            self.codebook = np.random.normal(loc=0.0, scale=1.0, size=(self.codebook_size,VectorLength,12)) + 1j * np.random.normal(loc=0.0, scale=1.0, size=(self.codebook_size,VectorLength,12))
            max_abs_values = np.abs(self.codebook).max(axis=1, keepdims=True)

            #The channel gain will in nearly all cases be ranging in magnitude from 0 to 1 so I normalize the maximum magnitude of each codebook vector to mag 1. 
            #self.codebook = self.codebook / max_abs_values
            
            #normalizing the vector magnitudes
            self.codebook=self.codebook /np.linalg.norm(self.codebook, axis=1, keepdims=True)
        
        self.num_tx_streams = num_tx_streams
        self.architecture = architecture # 'baseline', 'dMIMO_phase1', 'dMIMO_phase3_SU_MIMO'
        self.wideband = wideband
        
        snr_linear = 10**(snrdb/10)
        self.snr_linear = np.mean(snr_linear)

        self.num_BS_Ant = 4
        self.num_UE_Ant = 2

    def call(self, h_est, return_codebook=False):
        
        if self.method == '5G' and self.architecture == 'baseline':

            codebook = self.cal_codebook(h_est)
            PMI, rate_for_selected_precoder, precoding_matrices = self.cal_PMI(codebook, h_est)
            
            CSI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]
        
        elif self.method == '5G' and self.architecture == 'dMIMO_phase1':

            num_rx_ues = int(h_est.shape[2]/self.num_UE_Ant)

            PMI = []
            rate_for_selected_precoder = []
            precoding_matrices = []

            for rx_ue_idx in range(num_rx_ues):

                rx_ue_ant_idx = np.arange(rx_ue_idx*self.num_UE_Ant, (rx_ue_idx+1)*self.num_UE_Ant)
                codebook = self.cal_codebook(tf.gather(h_est, rx_ue_ant_idx, axis=2))
                PMI_temp, rate_for_selected_precoder_temp, precoding_matrices_temp = self.cal_PMI(codebook, tf.gather(h_est, rx_ue_ant_idx, axis=2))
                PMI.append(PMI_temp)
                rate_for_selected_precoder.append(rate_for_selected_precoder_temp)
                precoding_matrices.append(precoding_matrices_temp)
            
            PMI = np.asarray(PMI)
            rate_for_selected_precoder = np.asarray(rate_for_selected_precoder)
            precoding_matrices = np.asarray(precoding_matrices)
            
            CSI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]

        elif self.method == '5G' and self.architecture == 'dMIMO_phase3_SU_MIMO':

            self.nfft = h_est.shape[-1]

            num_ues = int(h_est.shape[4]/self.num_UE_Ant)
            PMI = []
            rate_for_selected_precoder = []
            precoding_matrices = []

            for ue_idx in range(num_ues):
                ue_ant_idx = np.arange(ue_idx*self.num_UE_Ant, (ue_idx+1)*self.num_UE_Ant)
                codebook = self.cal_codebook(tf.gather(h_est, ue_ant_idx, axis=4))
                PMI_temp, rate_for_selected_precoder_temp, precoding_matrices_temp = self.cal_PMI(codebook, tf.gather(h_est, ue_ant_idx, axis=4))
                PMI.append(PMI_temp)
                rate_for_selected_precoder.append(rate_for_selected_precoder_temp)
                precoding_matrices.append(precoding_matrices_temp)
            
            PMI = np.asarray(PMI)
            rate_for_selected_precoder = np.asarray(rate_for_selected_precoder)
            precoding_matrices = np.asarray(precoding_matrices)
            
            CSI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]
        
        elif self.method == 'RVQ':
            CSI_feedback_report  = self.VectorQuantizationLoader(h_est)
        else:
            raise Exception(f"The {self.method} CSI feedback mechanism has not been implemented. The simulator supports 5G standard CSI feedback and RVQ CSI feedback only.")
        
        if return_codebook:
            return CSI_feedback_report, codebook
        else:
            return CSI_feedback_report
    
    def cal_PMI(self, codebook, h_est):
        
        h_est = self.rb_mapper(h_est)

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]

        # num_rx_nodes = int((N_r - self.num_BS_Ant)/self.num_UE_Ant) + 1

        A_info = 0.83
        B_info = 0.73

        if self.architecture == 'baseline':

            num_codebook_elements = np.product(codebook.shape[:-2])
            codebook = codebook.reshape(-1, codebook.shape[-2], codebook.shape[-1])

            per_precoder_rate = np.zeros((h_est.shape[-1],num_codebook_elements))

            PMI = np.zeros((h_est.shape[-1]),dtype=int)
            rate_for_selected_precoder = np.zeros((h_est.shape[-1]))
            
            for codebook_idx in range(num_codebook_elements):

                h_eff = self.calculate_effective_channel(h_est, codebook[codebook_idx,...])

                snr_linear = np.sum(self.snr_linear)
                n_var = self.cal_n_var(h_eff, snr_linear)

                mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)
                mmse_inv  = mmse_inv + n_var*tf.eye(mmse_inv.shape[-1], dtype=mmse_inv.dtype)
                mmse_inv = tf.linalg.inv(mmse_inv)
                mmse_inv = tf.matmul(h_eff, mmse_inv, adjoint_a=True)

                per_stream_sinr = self.compute_sinr(h_eff, mmse_inv, n_var)

                avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                curr_codebook_rate = A_info * np.log2(1 + B_info * avg_sinr)
                per_precoder_rate[:, codebook_idx] = np.sum(curr_codebook_rate, axis=-1)

            precoding_matrices = np.zeros((h_est.shape[-1], codebook.shape[1], codebook.shape[2]), dtype=complex)

            for n in range(h_est.shape[-1]):
                PMI[n] = np.where(per_precoder_rate[n, :] == np.max(per_precoder_rate[n, :]))[0][0]
                rate_for_selected_precoder[n] = per_precoder_rate[n, PMI[n]]
                precoding_matrices[n, ...] = codebook[PMI[n]]
            
            precoding_matrices = precoding_matrices[np.newaxis, ...]

        elif self.architecture == 'dMIMO_phase1':

            num_codebook_elements = np.product(codebook.shape[:-2])
            codebook = codebook.reshape(-1, codebook.shape[-2], codebook.shape[-1])

            if self.codebook_selection_method == 'rate':

                per_precoder_rate = np.zeros((h_est.shape[-1],num_codebook_elements))

                PMI = np.zeros((h_est.shape[-1]),dtype=int)
            
                for codebook_idx in range(num_codebook_elements):

                    h_eff = self.calculate_effective_channel(h_est, codebook[codebook_idx,...])

                    snr_linear = np.sum(self.snr_linear)
                    n_var = self.cal_n_var(h_eff, snr_linear)

                    mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)
                    mmse_inv  = mmse_inv + n_var*tf.eye(mmse_inv.shape[-1], dtype=mmse_inv.dtype)
                    mmse_inv = tf.linalg.inv(mmse_inv)
                    mmse_inv = tf.matmul(h_eff, mmse_inv, adjoint_a=True)
                    
                    per_stream_sinr = self.compute_sinr(h_eff, mmse_inv, n_var)

                    avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                    curr_codebook_rate = A_info * np.log2(1 + B_info * avg_sinr)
                    per_precoder_rate[:, codebook_idx] = np.sum(curr_codebook_rate, axis=-1)

                if self.wideband:
                    precoding_matrices = np.zeros((1, codebook.shape[1], codebook.shape[2]), dtype=complex)
                    for n in range(h_est.shape[-1]):
                        PMI[n] = np.where(per_precoder_rate[n, :] == np.max(per_precoder_rate[n, :]))[0][0]
                    unique, counts = np.unique(PMI, return_counts=True)
                    PMI = unique[np.argmax(counts)]
                    rate_for_selected_precoder = np.mean(per_precoder_rate[:, PMI])
                    precoding_matrices[0, ...] = codebook[PMI]
                else:
                    precoding_matrices = np.zeros((h_est.shape[-1], codebook.shape[1], codebook.shape[2]), dtype=complex)
                    rate_for_selected_precoder = np.zeros((h_est.shape[-1]))
                    for n in range(h_est.shape[-1]):
                        PMI[n] = np.where(per_precoder_rate[n, :] == np.max(per_precoder_rate[n, :]))[0][0]
                        rate_for_selected_precoder[n] = per_precoder_rate[n, PMI[n]]
                        precoding_matrices[n, ...] = codebook[PMI[n]]
                
                precoding_matrices = precoding_matrices[np.newaxis, ...]
            elif self.codebook_selection_method == 'chordal_dist':

                B, _, N_r, _, N_t, N_sym, N_sc = h_est.shape
                Ns = self.num_tx_streams
                Nsb = N_sym * N_sc
                h_est_rs = tf.reshape(h_est, np.asarray([B, -1, N_r, N_t]))

                per_precoder_dist = np.zeros((Nsb, num_codebook_elements))
                PMI = np.zeros((Nsb,), dtype=int)

                vH_all = np.empty((Nsb, N_t, Ns), dtype=complex)

                for sb in range(Nsb):
                    _, _, vh = np.linalg.svd(h_est_rs[0, sb], full_matrices=False)
                    vH_all[sb, ...] = vh[:Ns].T

                for codebook_idx in range(num_codebook_elements):

                    w_unit = codebook[codebook_idx] / np.linalg.norm(codebook[codebook_idx], axis=0, keepdims=True)

                    vH_conjT = np.conj(vH_all).transpose(0, 2, 1)
                    proj = vH_conjT @ w_unit

                    frob_sq = np.sum(np.abs(proj) ** 2, axis=(1, 2))
                    per_precoder_dist[:, codebook_idx] = 1.0 - frob_sq / Ns

                if self.wideband:
                    precoding_matrices = np.zeros((1,
                                                codebook.shape[1],
                                                codebook.shape[2]), dtype=complex)

                    for sb in range(Nsb):
                        PMI[sb] = np.argmin(per_precoder_dist[sb, :])

                    unique, counts = np.unique(PMI, return_counts=True)
                    PMI = unique[np.argmax(counts)]                     # mode over RBs
                    precoding_matrices[0, ...] = codebook[PMI]

                else:
                    precoding_matrices = np.zeros((h_est.shape[-1],
                                                codebook.shape[1],
                                                codebook.shape[2]), dtype=complex)
                    for sb in range(Nsb):
                        PMI[sb] = np.argmin(per_precoder_dist[sb, :])
                        precoding_matrices[sb, ...] = codebook[PMI[sb]]

                precoding_matrices = precoding_matrices[np.newaxis, ...]
                rate_for_selected_precoder = None
            
        elif self.architecture == 'dMIMO_phase3_SU_MIMO':

            num_codebook_elements = np.product(codebook.shape[:-2])
            codebook = codebook.reshape(-1, codebook.shape[-2], codebook.shape[-1])

            if self.codebook_selection_method == 'rate':

                per_precoder_rate = np.zeros((h_est.shape[-1],num_codebook_elements))

                PMI = np.zeros((h_est.shape[-1]),dtype=int)
            
                for codebook_idx in range(num_codebook_elements):

                    h_eff = self.calculate_effective_channel(h_est, codebook[codebook_idx,...])

                    snr_linear = np.sum(self.snr_linear)
                    n_var = self.cal_n_var(h_eff, snr_linear)

                    mmse_inv = tf.matmul(h_eff, h_eff, adjoint_b=True)
                    mmse_inv  = mmse_inv + n_var*tf.eye(mmse_inv.shape[-1], dtype=mmse_inv.dtype)
                    mmse_inv = tf.linalg.inv(mmse_inv)
                    mmse_inv = tf.matmul(h_eff, mmse_inv, adjoint_a=True)
                    
                    per_stream_sinr = self.compute_sinr(h_eff, mmse_inv, n_var)

                    avg_sinr = self.eesm_average(per_stream_sinr, 0.25, 4)

                    curr_codebook_rate = A_info * np.log2(1 + B_info * avg_sinr)
                    per_precoder_rate[:, codebook_idx] = curr_codebook_rate

                if self.wideband:
                    precoding_matrices = np.zeros((1, codebook.shape[1], codebook.shape[2]), dtype=complex)
                    for n in range(h_est.shape[-1]):
                        PMI[n] = np.where(per_precoder_rate[n, :] == np.max(per_precoder_rate[n, :]))[0][0]
                    unique, counts = np.unique(PMI, return_counts=True)
                    PMI = unique[np.argmax(counts)]
                    rate_for_selected_precoder = np.mean(per_precoder_rate[:, PMI])
                    precoding_matrices[0, ...] = codebook[PMI]
                else:
                    precoding_matrices = np.zeros((h_est.shape[-1], codebook.shape[1], codebook.shape[2]), dtype=complex)
                    rate_for_selected_precoder = np.zeros((h_est.shape[-1]))
                    for n in range(h_est.shape[-1]):
                        PMI[n] = np.where(per_precoder_rate[n, :] == np.max(per_precoder_rate[n, :]))[0][0]
                        rate_for_selected_precoder[n] = per_precoder_rate[n, PMI[n]]
                        precoding_matrices[n, ...] = codebook[PMI[n]]
                
                precoding_matrices = precoding_matrices[np.newaxis, ...]

        return [PMI, rate_for_selected_precoder, precoding_matrices]

    def calculate_effective_channel(self, h_est, precoding_matrix):
    
        h_est_reshaped = tf.transpose(h_est, [0, 1, 3, 5, 6, 2, 4])
        h_est_reshaped = tf.cast(h_est_reshaped, dtype=precoding_matrix.dtype)

        h_eff = tf.matmul(h_est_reshaped, precoding_matrix)

        return h_eff
    
    def cal_n_var(self, h_eff, snr_linear):
        
        prod = tf.matmul(h_eff, h_eff, adjoint_b=True)
        sig_pow = np.abs(np.mean(np.trace(prod, axis1=-2, axis2=-1)))

        n_var = sig_pow / snr_linear

        return n_var
    
    def compute_sinr(self, h_eff, mmse_inv, n_var, eps=1e-12):
        """
        Per-stream SINR after linear MMSE combining.

        Inputs:
        h_eff    : [..., N_rx, N_s]   effective channel (antennas x streams)
        mmse_inv : [..., N_s, N_rx]   MMSE combiner W = H^H (H H^H + n I)^{-1}
        n_var    : scalar (or broadcastable) real noise variance

        Returns:
        sinr     : [..., N_s]         post-combiner SINR per stream
        """
        H = tf.convert_to_tensor(h_eff)
        W = tf.convert_to_tensor(mmse_inv)

        # Post-combiner effective channel S = W H  => [..., N_s, N_s]
        S = tf.matmul(W, H)

        # Desired signal power |S_ii|^2
        signal = tf.abs(tf.linalg.diag_part(S)) ** 2                      # [..., N_s]

        # Total power on each output stream: sum_j |S_ij|^2
        total = tf.reduce_sum(tf.abs(S) ** 2, axis=-1)                    # [..., N_s]

        # Multi-stream interference = total - desired
        interf = total - signal                                           # [..., N_s]

        # Noise term: n_var * ||W_i||^2  (row-wise squared norms of W)
        W_row_norm2 = tf.reduce_sum(tf.abs(W) ** 2, axis=-1)              # [..., N_s]
        noise = tf.cast(n_var, W.dtype.real_dtype) * W_row_norm2          # [..., N_s]

        # SINR_i = |S_ii|^2 / (sum_{j≠i} |S_ij|^2 + n_var * ||W_i||^2)
        sinr = signal / (interf + noise + tf.cast(eps, signal.dtype))
        
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

        N = int(np.size(sinr) / (sinr.shape[-1] * sinr.shape[-3]))
        exp_sum = np.sum(np.exp(-sinr / beta), axis=(0,1,2,3,5))
        exp_sum = 1 / N * exp_sum

        if np.any(exp_sum == 0):
            eesm_avg_sinr = np.mean(sinr, axis=(0,1,2,3,5))
        else:
            eesm_avg_sinr = -beta * np.log(exp_sum)

        return eesm_avg_sinr

    def cal_codebook(self, h_est):
        """
        Computes PMI codebook for 2/4 tx antennas
        Consult 3GPP TS 38.214 Section 5 for details
        """

        N_t = h_est.shape[4]
        P_CSI_RS = N_t

        if N_t == 2:
            # i_2 ∈ {0,1,2,3} gives the per-antenna relative phase φ_n = exp(jπ n / 2)
            i_11 = np.arange(0, self.N_1 * self.O_1)
            i_12 = np.arange(0, self.N_2 * self.O_2)
            i_2  = np.arange(0, 4)  # phase sweep like your 4-TX code

            l_all = i_11
            m_all = i_12
            n_all = i_2

            if self.num_tx_streams == 1:
                # W: [l, m, n, N_t, 1]
                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, 1), dtype=complex)

                for l in l_all:
                    for m in m_all:
                        v_lm = self.compute_v_lm(l, m)  # shape (2,1) when N_t=2
                        for n in n_all:
                            phi_n = np.exp(1j * np.pi * n / 2)
                            # Apply per-antenna relative phase [1, φ_n]^T ⊙ v_lm
                            d = np.array([[1.0], [phi_n]], dtype=complex)   # (2,1)
                            W[l, m, n, :, 0] = (v_lm * d).ravel()

                W = 1 / np.sqrt(P_CSI_RS) * W

            elif self.num_tx_streams == 2:
                # Use the same (l, m) and a frequency shift (k1, k2) like your 4-TX 2-stream case
                i_13 = np.arange(0, 2)
                k_1 = np.array((0, self.O_1))
                k_2 = np.array((0, 0))

                # W: [l, m, i13, n, N_t, 2]
                W = np.zeros((len(l_all), len(m_all), len(i_13), len(n_all), N_t, 2), dtype=complex)

                for l in l_all:
                    for m in m_all:
                        v_lm = self.compute_v_lm(l, m)  # (2,1)
                        for i_13_idx in i_13:
                            l_ = l + k_1[i_13_idx]
                            m_ = m + k_2[i_13_idx]
                            v_lm_shift = self.compute_v_lm(l_, m_)  # (2,1)

                            for n in n_all:
                                phi_n = np.exp(1j * np.pi * n / 2)
                                # Columns mirror your 4-TX sign/phase pattern:
                                # col1 = [1,  +φ_n]^T ⊙ v_lm
                                # col2 = [1,  -φ_n]^T ⊙ v_lm_shift
                                d_pos = np.array([[1.0], [phi_n]], dtype=complex)
                                d_neg = np.array([[1.0], [-phi_n]], dtype=complex)

                                col1 = (v_lm * d_pos).reshape(N_t, 1)         # (2,1)
                                col2 = (v_lm_shift * d_neg).reshape(N_t, 1)    # (2,1)
                                W[l, m, i_13_idx, n, :, :] = np.hstack((col1, col2))

                W = 1 / np.sqrt(2 * P_CSI_RS) * W

            else:
                raise Exception(f"5G standard PMI feedback for {N_t} tx and {self.num_tx_streams} streams not implemented. Supported for 1–2 streams when N_t=2.")
        elif N_t == 4:

            if self.num_tx_streams == 1:
                
                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_2 = np.arange(0,4)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:

                        v_lm = self.compute_v_lm(l, m)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)

                            W[l,m,n,...] = np.vstack((v_lm, phi_n * v_lm))

                W = 1/np.sqrt(P_CSI_RS) * W

            elif self.num_tx_streams == 2:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                i_13 = np.arange(0,2)
                k_1 = np.array((0, self.O_1))
                k_2 = np.array((0, 0))
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(i_13), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:
                        for i_13_idx in i_13:
                            

                            l_ = l + k_1[i_13_idx]
                            m_ = m + k_2[i_13_idx]

                            v_lm = self.compute_v_lm(l, m)
                            v_l_m_ = self.compute_v_lm(l_, m_)
                            
                            for n in n_all:

                                phi_n = np.exp(1j * np.pi * n / 2)
                                
                                col_1 = np.vstack((v_lm, phi_n * v_lm))
                                col_2 = np.vstack((v_l_m_, -phi_n * v_l_m_))
                                W[l,m,i_13_idx,n,...] = np.hstack((col_1, col_2))
                
                W = 1/np.sqrt(2 * P_CSI_RS) * W
            
            elif self.num_tx_streams == 3:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                k_1 = self.O_1
                k_2 = 0
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:                            

                        l_ = l + k_1
                        m_ = m + k_2

                        v_lm = self.compute_v_lm(l, m)
                        v_l_m_ = self.compute_v_lm(l_, m_)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)
                            
                            col_1 = np.vstack((v_lm, phi_n * v_lm))
                            col_2 = np.vstack((v_l_m_, phi_n * v_l_m_))
                            col_3 = np.vstack((v_lm, -phi_n * v_lm))
                            W[l,m,n,...] = np.hstack((col_1, col_2, col_3))
                
                W = 1/np.sqrt(3 * P_CSI_RS) * W
            
            elif self.num_tx_streams == 4:

                i_11 = np.arange(0, self.N_1 * self.O_1)
                i_12 = np.arange(0, self.N_2 * self.O_2)
                k_1 = self.O_1
                k_2 = 0
                i_2 = np.arange(0,2)

                l_all = i_11
                m_all = i_12
                n_all = i_2

                W = np.zeros((len(l_all), len(m_all), len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for l in l_all:
                    for m in m_all:                            

                        l_ = l + k_1
                        m_ = m + k_2

                        v_lm = self.compute_v_lm(l, m)
                        v_l_m_ = self.compute_v_lm(l_, m_)
                        
                        for n in n_all:

                            phi_n = np.exp(1j * np.pi * n / 2)
                            
                            col_1 = np.vstack((v_lm, phi_n * v_lm))
                            col_2 = np.vstack((v_l_m_, phi_n * v_l_m_))
                            col_3 = np.vstack((v_lm, -phi_n * v_lm))
                            col_4 = np.vstack((v_l_m_, -phi_n * v_l_m_))
                            W[l,m,n,...] = np.hstack((col_1, col_2, col_3, col_4))
                
                W = 1/np.sqrt(4 * P_CSI_RS) * W
            else:
                raise Exception(f"5G standard PMI feedback for {self.num_tx_streams} spatial streams has not been implemented. The simulator supports 1-4 spatial streams only.")

        else:
            raise Exception(f"5G standard PMI feedback for {N_t} x {self.num_tx_streams} MIMO order has not been implemented. The simulator supports MIMO orders 4 tx antennas and 1-4 spatial streams only.")
        
        return W


    def compute_u_m(self, m):

        if self.N_2 == 1:
            u_m = 1
        elif self.N_2 > 1:
            u_m = np.exp((2j * np.pi * m * np.arange(0, self.N_2)) / (self.O_2 * self.N_2) )
        else:
            raise Exception(f"Incorrect choice of N_2")

        return u_m

    def compute_v_lm(self, l, m):

        u_m = self.compute_u_m(m)

        v_l = np.exp( (2j * np.pi * l * np.arange(0, self.N_1)) / (self.O_1 * self.N_1))

        v_l_m = np.outer(u_m, v_l).flatten()
        v_l_m = v_l_m.reshape(-1, 1)

        return v_l_m
    
    def quantize(self, data,i_RxNodeIndex):
        """
        Quantizes each data vector using multiple random vectors and stores the indices in binary format.
        Args:
            data (ndarray): Input data to compress (NxD matrix, complex).

        Returns:
            codebook (ndarray): The random codebook used for quantization (complex).
            binary_encoded_indices (list of str): List of binary representations of the indices for each data vector.
        """
        binary_encoded_indices = []
        residuals = data
        binary_index = ""
        
        #This is incase we ever had a case where we want to feedback two seperate codewords rather than only one with all X bits. 
        #Generally it will be that all bits are used for a single codebook index to increase the number of codebook entries. 
        for j in range(self.num_codewords):

            residuals_squeezed = tf.squeeze(tf.cast(residuals,dtype=tf.complex128))
            
            dot_products = tf.abs(tf.tensordot(residuals_squeezed, self.codebook[:,:,i_RxNodeIndex], axes=[[0], [1]]))

            # Compute the norms in one operation
            residual_norm = tf.norm(residuals_squeezed)
            codebook_norms = tf.norm(self.codebook[:,:,i_RxNodeIndex], axis=1)

            # Calculate the cosine distances
            distances = tf.abs(dot_products / (tf.abs(residual_norm) * tf.abs(codebook_norms)))

            #Select best distance 
            best_index = int(tf.argmax(distances))

            #Update the residual
            residuals = residuals - self.codebook[best_index,:,i_RxNodeIndex]

            # Convert the index to binary representation (padded to the specified number of bits)
            binary_codeword = format(best_index, f'0{self.bits_per_codeword}b')
            binary_index += binary_codeword  # Concatenate binary codewords

            binary_encoded_indices.append(binary_index)

        return binary_encoded_indices

    def VectorQuantizationLoader(self,H):
        #Segments the channel into chunks then passes the chunks to the codebook quantizer
        
        #Index the channel into chunks then feed into the random vector_quantization
        #define the array of the right size then make the nested list
        binary_encoded_indices=np.zeros((H.shape[0],1,int((H.shape[2]/2)),14,int(np.round(H.shape[6]/12))))
        binary_string_indices = [[[[['' for _ in range(binary_encoded_indices.shape[4])]
                            for _ in range(binary_encoded_indices.shape[3])]
                            for _ in range(binary_encoded_indices.shape[2])]
                            for _ in range(binary_encoded_indices.shape[1])]
                            for _ in range(binary_encoded_indices.shape[0])]
        
        #Store this for use in the reconstruction, it is just used to make sure the reconstructed channel has the same shape. 
        self.H=H
        
        #For each subframe
        for i_BatchIndex in range(H.shape[0]):
            #For each symbol
            for i_SymbolIndex in range(H.shape[5]):
                #For each RX Node
                for i_RxNodeIndex in np.arange(0,H.shape[2],2):
                    #Segment the channel into all TX to a particular RX Node
                    
                    H_perRxNode=np.squeeze(H[i_BatchIndex,0,i_RxNodeIndex:i_RxNodeIndex+2,0,:,i_SymbolIndex,:])
                    
                    # For each RBG we produce one quantized value.     
                    for i_RBG in np.arange(0,H.shape[6],12):
                        #Accounts for the fact that we feedback 512 but 512/12 is not divisible so for the final RBG we take only what is left
                        if i_RBG+12 > H_perRxNode.shape[2]:
                            H_RBG=H_perRxNode[:,:,i_RBG:]
                        else:
                            H_RBG=H_perRxNode[:,:,i_RBG:i_RBG+12]
                            
                        #Take mean over the RBG
                        H_RBG=np.mean(H_RBG,2)
                        
                        #Reshape the final per node channels into one vector. 
                        H_RBGVector=np.reshape(H_RBG,(-1,1))

                        #Normalize the channel to the same normalization used for the codebook
                        H_RBGVector=H_RBGVector/np.linalg.norm(H_RBGVector, axis=0, keepdims=True)
                        
                        #Quantizing the complex vector and storing in a list
                        binary_string_indices[i_BatchIndex][0][int(i_RxNodeIndex/2)][i_SymbolIndex][int(i_RBG/12)] = self.quantize(H_RBGVector,int(i_RxNodeIndex/2))
        
        return binary_string_indices
    
    def Reconstruction(self,binary_string_indices):
        #Define the full matrix we will construct 
        H_Reconstructed=np.zeros_like(self.H)
        
        #For each subframe
        for i_BatchIndex in range(self.H.shape[0]):
            #For each symbol
            for i_SymbolIndex in range(self.H.shape[5]):
                #For each RX Node
                for i_RxNodeIndex in np.arange(0,self.H.shape[2],2):
                    #Segment the channel into all TX to a particular RX Node
                    
                    for i_RBG in np.arange(0,self.H.shape[6],12):
                        #repeats incase we seperate the total bits into multiple codewords. we normally dont but I have the functionality.
                        for i_NumCodewords in range(self.num_codewords):
                            #Pull the particular binary string
                            BinaryString=binary_string_indices[i_BatchIndex][0][int(i_RxNodeIndex/2)][i_SymbolIndex][int(i_RBG/12)]
                            #Process and strip the string and conver it to integer codebook index
                            BinaryString=str(BinaryString[(self.bits_per_codeword*i_NumCodewords):(self.bits_per_codeword*i_NumCodewords+1)])
                            Codebook_Index=int(BinaryString.strip("[]'\""),2)
                            
                            #Pull the correct index and reshape into the correct channel shape
                            H_Vector=self.codebook[Codebook_Index,:,int(i_RxNodeIndex/2)]
                            H_PerNode=np.reshape(H_Vector,(2,self.H.shape[4]))
                            
                            #Handling for non divisble subcarriers. 
                            if (i_RBG+1)+12 > self.H.shape[6]:
                                H_Reconstructed[i_BatchIndex,0,i_RxNodeIndex:i_RxNodeIndex+2,0,:,i_SymbolIndex,i_RBG:]+=np.tile(np.expand_dims(H_PerNode, axis=-1), (1, 1, self.H.shape[6]-(i_RBG)))
                            else:
                                H_Reconstructed[i_BatchIndex,0,i_RxNodeIndex:i_RxNodeIndex+2,0,:,i_SymbolIndex,i_RBG:(i_RBG+12)]+=np.tile(np.expand_dims(H_PerNode, axis=-1), (1, 1, 12))
                
                        
        
        return H_Reconstructed


    def reconstruct_channel(self, precoding_matrices, snr_assumed_dBm, n_var, bs_txpwr_dbm):
        
        rx_sig_pow = n_var * 10**(snr_assumed_dBm/10)
        tx_sig_pow = 10**(bs_txpwr_dbm/10)
        s = np.sqrt(rx_sig_pow / tx_sig_pow)

        # [num_rx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        h_freq_csi_reconstructed = precoding_matrices * s

        # [batch_size, num_rx, num_streams_per_rx, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        reshaped_array = h_freq_csi_reconstructed.transpose(0, 3, 2, 1)
        reshaped_array = reshaped_array[:, np.newaxis, :, np.newaxis, :, np.newaxis, :]
        repeated_array = np.repeat(reshaped_array, 14, axis=5)
        h_freq_csi_reconstructed = tf.convert_to_tensor(repeated_array)

        if self.architecture == 'baseline':
            padding = self.num_BS_Ant - h_freq_csi_reconstructed.shape[2]
        elif self.architecture == 'dMIMO_phase1':
            padding = self.num_UE_Ant - h_freq_csi_reconstructed.shape[2]
        else:
            padding = 0

        padding_mask = [
            [0, 0],  # No padding on the 1st dimension
            [0, 0],  # No padding on the 2nd dimension
            [0, padding],  # Pad the 3rd dimension from 2 to 4 (2 zeros after)
            [0, 0],  # No padding on the 4th dimension
            [0, 0],  # No padding on the 5th dimension
            [0, 0],  # No padding on the 6th dimension
            [0, 0],  # No padding on the 7th dimension
        ]

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq_csi_reconstructed = tf.pad(h_freq_csi_reconstructed, padding_mask)

        return h_freq_csi_reconstructed

    def rb_mapper(self, H):

        num_full_rbs = self.nfft // self.subcarriers_per_RB
        remainder_subcarriers = self.nfft % self.subcarriers_per_RB

        # Initialize an array to store the averaged RBs
        if remainder_subcarriers > 0:
            rb_data = np.zeros(H.shape[:-1].concatenate(num_full_rbs + 1), dtype=complex)
        else:
            rb_data = np.zeros(H.shape[:-1].concatenate(num_full_rbs), dtype=complex)

        # Compute mean across each full RB
        for rb in range(num_full_rbs):
            start = rb * self.subcarriers_per_RB
            end = start + self.subcarriers_per_RB
            rb_data[..., rb] = np.mean(H[..., start:end], axis=-1)

        # Calculate the mean for the remaining subcarriers
        if remainder_subcarriers > 0:
            rb_data[..., -1] = np.mean(H[..., -remainder_subcarriers:], axis=-1)

        demapped_H = tf.repeat(rb_data, repeats=np.ceil(self.nfft/rb_data.shape[-1]), axis=-1)
        demapped_H = demapped_H[..., :self.nfft]
        
        return demapped_H
    


class RandomVectorQuantizer(Layer):
    def __init__(self, bits_per_codeword, vector_dim, seed=42, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.bits_per_codeword = bits_per_codeword
        self.vector_dim = vector_dim
        self.codebook_size = 2 ** bits_per_codeword

        # Generate random codebook
        self.codebook = self.generate_codebook(seed) # shape: (codebook_size, vector_dim)

    def generate_codebook(self, seed):
        rng = tf.random.Generator.from_seed(seed)
        # Create a random codebook with normalized complex vectors
        codebook = tf.complex(rng.normal(shape=(self.codebook_size, self.vector_dim)),
                                 rng.normal(shape=(self.codebook_size, self.vector_dim)))
        codebook /= tf.linalg.norm(codebook, axis=1, keepdims=True)
        return codebook

    def __call__(self, inputs:tf.Tensor) -> tf.Tensor:
        """
        Quantizes each input vector using the random codebook and returns the indices if inputs.dtype==complex.
        Otherwise if inputs.dtype==int, it reconstructs the vectors from the codebook using the provided indices.
        Args:
            inputs (Tensor): Input data to quantize shape= (..., vector_dim) if input dtype=complex or (...,) if input dtype=int.
        Returns:
            indices (Tensor): Indices of the quantized vectors of shape (...,) and dtype=int or
            reconstructed_vectors (Tensor): Reconstructed vectors from the codebook using the provided indices of shape (..., vector_dim) and dtype=complex.
        """
        return super().__call__(inputs)
    
    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        if inputs.dtype == tf.complex64 or inputs.dtype == tf.complex128:
            # Quantization process
            dot_products = tf.math.abs(tf.tensordot(tf.math.conj(inputs), self.codebook, axes=[[ -1], [1]])) # shape: (..., codebook_size)
            indices = tf.argmax(dot_products, axis=-1) # shape: (...,)
            return indices
        else:
            # Reconstruction process
            return tf.gather(self.codebook, inputs, axis=0) # shape: (..., vector_dim)
        

class RandomVectorQuantizerNumpy:
    """
    Numpy equivalent of the TF RandomVectorQuantizer.
    - If called with complex/float vectors of shape (..., vector_dim) it returns
      indices of the nearest codebook vectors with shape (...,) (dtype=int64).
    - If called with integer indices of shape (...,) it returns reconstructed
      complex vectors with shape (..., vector_dim).
    """
    def __init__(self, bits_per_codeword, vector_dim, seed=42):
        self.bits_per_codeword = int(bits_per_codeword)
        self.vector_dim = int(vector_dim)
        self.codebook_size = 2 ** self.bits_per_codeword
        self.codebook = self._generate_codebook(seed)  # shape: (codebook_size, vector_dim)

    def _generate_codebook(self, seed) -> np.ndarray:
        rng = np.random.default_rng(seed)
        real = rng.normal(size=(self.codebook_size, self.vector_dim))
        imag = rng.normal(size=(self.codebook_size, self.vector_dim))
        cb = real + 1j * imag
        norms = np.linalg.norm(cb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cb = cb / norms
        return cb.astype(np.complex64)

    def __call__(self, inputs):
        return self.quantize_or_reconstruct(inputs)

    def quantize_or_reconstruct(self, inputs):
        arr = np.asarray(inputs)
        
        if np.iscomplexobj(arr) or np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.complex64)
            arr_shape = arr.shape
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(-1, arr_shape[-1])
            assert arr.shape[-1] == self.vector_dim, f"Expected last dim {self.vector_dim}, got {arr.shape[-1]}"
            
            dots = _parallel_dot(arr, self.codebook)  # This is parallel across CPU cores
            idx = np.argmax(dots, axis=-1).astype(np.int64)
            idx = np.reshape(idx, arr_shape[:-1])
            return idx
        else:
            idx = arr.astype(np.int64)
            if np.any((idx < 0) | (idx >= self.codebook_size)):
                raise IndexError("RandomVectorQuantizerNumpy.quantize_or_reconstruct: Index out of range")
            return self.codebook[idx]
        

from numba import njit, prange

@njit(parallel=True)
def _parallel_dot(arr, codebook):
    """
    arr: (batch_size, vector_dim) complex64
    codebook: (codebook_size, vector_dim) complex64
    returns: (batch_size, codebook_size) float32 abs inner products
    """
    batch_size = arr.shape[0]
    codebook_size = codebook.shape[0]
    
    dots = np.zeros((batch_size, codebook_size), dtype=np.float32)
    
    for i in prange(batch_size):
        # conj(arr[i]) @ codebook[j] for all j
        dots[i] = np.abs(arr[i].conj() @ codebook.T)
    
    return dots