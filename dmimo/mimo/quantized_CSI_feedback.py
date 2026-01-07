import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Layer

class quantized_CSI_feedback(Layer):
    """CSI feedback report generation"""

    def __init__(self,
                method,
                codebook_selection_method,
                num_tx_streams,
                architecture,
                snrdb,
                L=None,
                N_1=None,
                N_2=1,
                O_1=4,
                O_2=1,
                wideband=False,
                rbs_per_subband=4,
                total_bits=None,
                VectorLength=None,
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.real_dtype = dtype.real_dtype

        self.nfft = 512
        self.subcarriers_per_RB = 12

        self.codebook_selection_method = codebook_selection_method # 'rate', 'chordal_dist' for type I, None for type II

        self.method = method
        if "phase2" not in architecture:
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
        else:
            self.L = L # Number of beams to select
            self.N_1 = N_1 # Number of quantization points in the horizontal dimension
            self.N_2 = N_2 # Number of quantization points in the vertical dimension
            self.O_1 = O_1 # Horizontal oversampling factor
            self.O_2 = O_2 # Vertical oversampling factor
        
        self.num_tx_streams = num_tx_streams
        self.architecture = architecture # 'baseline', 'dMIMO_phase1', 'dMIMO_phase3_SU_MIMO'
        self.wideband = wideband

        self.rbs_per_subband = rbs_per_subband
        
        snr_linear = 10**(snrdb/10)
        self.snr_linear = np.mean(snr_linear)

        self.num_BS_Ant = 4
        self.num_UE_Ant = 2

    def call(
        self,
        h_est,
        return_codebook=False,
        return_feedback_bits=False,
        w1_beam_indices_override=None,
        return_components=False,
    ):
        
        if self.method == '5G' and self.architecture == 'baseline':

            codebook = self.cal_codebook_type_I(h_est)
            PMI, rate_for_selected_precoder, precoding_matrices = self.cal_PMI_type_I(codebook, h_est)
            
            PMI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]

        elif self.method == '5G' and self.architecture == 'dMIMO_phase2_rel_15_type_II':

            h_est = self.rb_mapper(h_est)
            h_est_rb = tf.gather(h_est, tf.range(0, self.nfft, self.subcarriers_per_RB), axis=-1)
            h_est_rb = tf.reduce_mean(h_est_rb, axis=-2, keepdims=False)
            h_est_rb = tf.squeeze(h_est_rb)
            h_est_rb = tf.cast(h_est_rb, self.dtype)

            if self.L is None:
                self.L = tf.reduce_min([4, h_est_rb.shape[1]])
            else:
                self.L = tf.reduce_min([self.L, h_est_rb.shape[1]])

            if self.N_1 is None:
                self.N_1 = h_est_rb.shape[1]

            V = self.build_sd_beam_grid()
            V = tf.convert_to_tensor(V, dtype=self.dtype)

            num_rx_ues = int((h_est.shape[2] - self.num_BS_Ant) /self.num_UE_Ant)

            num_streams_per_UE = self.num_tx_streams // (num_rx_ues + 2)

            W_rb_all = []

            for rx_idx in range(num_rx_ues+1):

                if rx_idx == 0:
                    # BS
                    rx_ant_idx = np.arange(0, self.num_BS_Ant)
                    Ns = 2 * num_streams_per_UE 
                else:
                    rx_ant_idx = np.arange((rx_idx-1)*self.num_UE_Ant + self.num_BS_Ant, (rx_idx)*self.num_UE_Ant + self.num_BS_Ant)
                    Ns = num_streams_per_UE

                assert Ns <= 2, "Type II codebook only defined for rank 1 and rank 2"

                curr_h_est = tf.gather(h_est_rb, rx_ant_idx, axis=0)

                W_rb = self.type_II_precoder(curr_h_est, V, Ns)

                W_rb_all.append(W_rb)

            
            h_est_quant = tf.concat(W_rb_all, axis=-1)  # [N_RB, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=np.ceil(self.nfft/W_rb.shape[0]), axis=0)
            h_est_quant = h_est_quant[:self.nfft, ...]
            h_est_quant = h_est_quant[tf.newaxis, ...]  # [1, N_fft, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=h_est.shape[5], axis=0)  # [N_syms, N_fft, N_tx, total_Ns]
            h_est_quant = tf.transpose(h_est_quant, perm=[3, 2, 0, 1])
            h_est_quant  = h_est_quant[tf.newaxis, tf.newaxis, :, tf.newaxis, :, :, :] # [1,1,total_Ns,1,N_tx,N_syms,N_fft]

            PMI_feedback_report = h_est_quant

        elif self.method == '5G' and self.architecture == 'dMIMO_phase2_type_II_CB1':

            h_est = self.rb_mapper(h_est)
            h_est_rb = tf.gather(h_est, tf.range(0, self.nfft, self.subcarriers_per_RB), axis=-1)
            h_est_rb = tf.reduce_mean(h_est_rb, axis=-2, keepdims=False)
            h_est_rb = tf.squeeze(h_est_rb)
            h_est_rb = tf.cast(h_est_rb, self.dtype)

            num_rx_ues = int((h_est.shape[2] - self.num_BS_Ant) /self.num_UE_Ant)

            num_tx_ues = int((h_est.shape[4] - self.num_BS_Ant) /self.num_UE_Ant)

            num_streams_per_UE = self.num_tx_streams // (num_rx_ues + 2)

            W_rb_all = []
            ref_tx = 0

            for rx_idx in range(num_rx_ues+1):

                if rx_idx == 0:
                    # BS
                    rx_ant_idx = np.arange(0, self.num_BS_Ant)
                    Ns = 2 * num_streams_per_UE 
                else:
                    rx_ant_idx = np.arange((rx_idx-1)*self.num_UE_Ant + self.num_BS_Ant, (rx_idx)*self.num_UE_Ant + self.num_BS_Ant)
                    Ns = num_streams_per_UE

                assert Ns <= 2, "Type II codebook only defined for rank 1 and rank 2"

                W_rb_all_tx = []                

                for tx_idx in range(num_tx_ues+1):
                    
                    if tx_idx == 0:
                        tx_ant_idx = np.arange(0, self.num_BS_Ant)
                    else:
                        tx_ant_idx = np.arange((tx_idx-1)*self.num_UE_Ant + self.num_BS_Ant, (tx_idx)*self.num_UE_Ant + self.num_BS_Ant)

                    curr_h_est = tf.gather(tf.gather(h_est_rb, rx_ant_idx, axis=0), tx_ant_idx, axis=1) # [N_r_k, N_t_g, N_rb]

                    if tx_idx == ref_tx:
                        curr_h_est_reshaped = tf.transpose(curr_h_est, [2, 0, 1])  # [N_RB, N_rx, N_tx]
                        sb_ids = tf.range(curr_h_est_reshaped.shape[0], dtype=tf.int32) // int(self.rbs_per_subband)
                        ref_h_est_sb = tf.math.segment_mean(curr_h_est_reshaped, sb_ids) # [N_rx, N_rx, N_tx]

                    # self.N_1 = curr_h_est.shape[1] // 2
                    N_1_g = curr_h_est.shape[1]
                    L_g = min(4, curr_h_est.shape[1])

                    V = self.build_sd_beam_grid(N1=N_1_g)
                    V = tf.convert_to_tensor(V, dtype=self.dtype)

                    W_rb = self.type_II_precoder(curr_h_est, V, Ns, L=L_g, N1=N_1_g)

                    W_rb_all_tx.append(W_rb)

                for tx_idx in range(num_tx_ues+1):

                    if tx_idx == 0:
                        tx_ant_idx = np.arange(0, self.num_BS_Ant)
                    else:
                        tx_ant_idx = np.arange((tx_idx-1)*self.num_UE_Ant + self.num_BS_Ant, (tx_idx)*self.num_UE_Ant + self.num_BS_Ant)

                    curr_h_est = tf.gather(tf.gather(h_est_rb, rx_ant_idx, axis=0), tx_ant_idx, axis=1) # [N_r_k, N_t_g, N_rb]

                    curr_h_est_reshaped = tf.transpose(curr_h_est, [2, 0, 1])  # [N_RB, N_rx, N_tx]
                    sb_ids = tf.range(curr_h_est_reshaped.shape[0], dtype=tf.int32) // int(self.rbs_per_subband)
                    curr_h_est_sb = tf.math.segment_mean(curr_h_est_reshaped, sb_ids) # [N_sb, N_rx, N_tx]

                    if tx_idx != ref_tx:
                        q = self.compute_q_per_sb(ref_h_est_sb, W_rb_all_tx[ref_tx], curr_h_est_sb, W_rb_all_tx[tx_idx])
                    
                        W_rb_all_tx[tx_idx] = q[:,tf.newaxis,tf.newaxis] * W_rb_all_tx[tx_idx]

                W_rb_all_tx = tf.concat(W_rb_all_tx, axis=1)  # [N_RB, N_tx_g, Ns_for_curr_rx]

                W_rb_all.append(W_rb_all_tx)

            h_est_quant = tf.concat(W_rb_all, axis=-1)  # [N_RB, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=np.ceil(self.nfft/W_rb.shape[0]), axis=0)
            h_est_quant = h_est_quant[:self.nfft, ...]
            h_est_quant = h_est_quant[tf.newaxis, ...]  # [1, N_fft, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=h_est.shape[5], axis=0)  # [N_syms, N_fft, N_tx, total_Ns]
            h_est_quant = tf.transpose(h_est_quant, perm=[3, 2, 0, 1])
            h_est_quant  = h_est_quant[tf.newaxis, tf.newaxis, :, tf.newaxis, :, :, :] # [1,1,total_Ns,1,N_tx,N_syms,N_fft]

            PMI_feedback_report = h_est_quant

        elif self.method == '5G' and self.architecture == 'dMIMO_phase2_type_II_CB2':

            h_est = self.rb_mapper(h_est)
            h_est_rb = tf.gather(h_est, tf.range(0, self.nfft, self.subcarriers_per_RB), axis=-1)
            h_est_rb = tf.reduce_mean(h_est_rb, axis=-2, keepdims=False)
            h_est_rb = tf.squeeze(h_est_rb)
            h_est_rb = tf.cast(h_est_rb, self.dtype)

            num_rx_ues = int((h_est.shape[2] - self.num_BS_Ant) /self.num_UE_Ant)

            num_tx_ues = int((h_est.shape[4] - self.num_BS_Ant) /self.num_UE_Ant)

            num_streams_per_UE = self.num_tx_streams // (num_rx_ues + 2)

            W_rb_all = []
            feedback_bits = []
            components = []

            rx_ue_ant_indices = [
                np.arange(0, 2),
                np.arange(2, 4),
            ]
            for rx_idx in range(num_rx_ues):
                rx_ue_ant_indices.append(
                    np.arange((rx_idx) * self.num_UE_Ant + self.num_BS_Ant, (rx_idx + 1) * self.num_UE_Ant + self.num_BS_Ant)
                )
            
            total_pmi_streams = 0

            Ns = num_streams_per_UE

            assert Ns <= 2, "Type II codebook only defined for rank 1 and rank 2"

            for rx_idx, rx_ant_idx in enumerate(rx_ue_ant_indices):

                rx_override = None
                if w1_beam_indices_override is not None and rx_idx < len(w1_beam_indices_override):
                    rx_override = w1_beam_indices_override[rx_idx]

                curr_h_est_big = tf.gather(h_est_rb, rx_ant_idx, axis=0)

                W1_rb_all_tx = []
                W1_indices = []        

                for tx_idx in range(num_tx_ues+1):
                    
                    if tx_idx == 0:
                        tx_ant_idx = np.arange(0, self.num_BS_Ant)
                    else:
                        tx_ant_idx = np.arange((tx_idx-1)*self.num_UE_Ant + self.num_BS_Ant, (tx_idx)*self.num_UE_Ant + self.num_BS_Ant)

                    curr_h_est = tf.gather(tf.gather(h_est_rb, rx_ant_idx, axis=0), tx_ant_idx, axis=1) # [N_r_k, N_t_g, N_rb]

                    curr_R = self.compute_tx_covariance(curr_h_est)

                    N_1_g = curr_h_est.shape[1]
                    L_g = min(4, curr_h_est.shape[1])

                    V = self.build_sd_beam_grid(N1=N_1_g)
                    V = tf.convert_to_tensor(V, dtype=self.dtype)

                    forced_indices = None
                    if rx_override is not None and isinstance(rx_override, (list, tuple)) and tx_idx < len(rx_override):
                        tx_override = rx_override[tx_idx]
                        if tx_override is not None:
                            forced_indices = tf.convert_to_tensor(tx_override, dtype=tf.int32)

                    if forced_indices is not None and tf.size(forced_indices) > 0:
                        col_idx_list = forced_indices
                    else:
                        _, _, col_idx_list = self.select_L_beams(curr_R, V, return_column_indices=True, L=L_g, N1=N_1_g)

                    W1_indices.append(col_idx_list)

                    W_1 = tf.gather(V, col_idx_list, axis=1)  # [Nt, L]

                    W1_rb_all_tx.append(W_1)

                W1_rb_all_tx = self.block_diag(W1_rb_all_tx)

                W_rb_out = self.type_II_precoder_CB2(
                    W1_rb_all_tx,
                    curr_h_est_big,
                    Ns,
                    N1=W1_rb_all_tx.shape[0],
                    return_feedback_bits=return_feedback_bits,
                    w1_beam_indices=W1_indices,
                    return_components=return_components,
                )

                if return_components:
                    if return_feedback_bits:
                        W_rb, pmi_bits, component_output = W_rb_out
                        feedback_bits.append(pmi_bits)
                    else:
                        W_rb, component_output = W_rb_out
                    components.append(component_output)
                else:
                    if return_feedback_bits:
                        W_rb, pmi_bits = W_rb_out
                        feedback_bits.append(pmi_bits)
                    else:
                        W_rb = W_rb_out

                W_rb_all.append(W_rb)
                total_pmi_streams += Ns

            assert total_pmi_streams == self.num_tx_streams, "Total PMI streams must match the configured number of transmit streams"

            h_est_quant = tf.concat(W_rb_all, axis=-1)  # [N_RB, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=np.ceil(self.nfft/W_rb.shape[0]), axis=0)
            h_est_quant = h_est_quant[:self.nfft, ...]
            h_est_quant = h_est_quant[tf.newaxis, ...]  # [1, N_fft, N_tx, total_Ns]
            h_est_quant = tf.repeat(h_est_quant, repeats=h_est.shape[5], axis=0)  # [N_syms, N_fft, N_tx, total_Ns]
            h_est_quant = tf.transpose(h_est_quant, perm=[3, 2, 0, 1])
            h_est_quant  = h_est_quant[tf.newaxis, tf.newaxis, :, tf.newaxis, :, :, :] # [1,1,total_Ns,1,N_tx,N_syms,N_fft]

            if return_components:
                if return_feedback_bits:
                    PMI_feedback_report = (h_est_quant, feedback_bits, components)
                else:
                    PMI_feedback_report = (h_est_quant, None, components)
            else:
                if return_feedback_bits:
                    PMI_feedback_report = (h_est_quant, feedback_bits)
                else:
                    PMI_feedback_report = (h_est_quant, None)
            

        elif self.method == '5G' and self.architecture == 'dMIMO_phase1':

            num_rx_ues = int(h_est.shape[2]/self.num_UE_Ant)

            PMI = []
            rate_for_selected_precoder = []
            precoding_matrices = []

            for rx_ue_idx in range(num_rx_ues):

                rx_ue_ant_idx = np.arange(rx_ue_idx*self.num_UE_Ant, (rx_ue_idx+1)*self.num_UE_Ant)
                codebook = self.cal_codebook_type_I(tf.gather(h_est, rx_ue_ant_idx, axis=2))
                PMI_temp, rate_for_selected_precoder_temp, precoding_matrices_temp = self.cal_PMI_type_I(codebook, tf.gather(h_est, rx_ue_ant_idx, axis=2))
                PMI.append(PMI_temp)
                rate_for_selected_precoder.append(rate_for_selected_precoder_temp)
                precoding_matrices.append(precoding_matrices_temp)
            
            PMI = np.asarray(PMI)
            rate_for_selected_precoder = np.asarray(rate_for_selected_precoder)
            precoding_matrices = np.asarray(precoding_matrices)
            
            PMI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]

        elif self.method == '5G' and self.architecture == 'dMIMO_phase3_SU_MIMO':

            self.nfft = h_est.shape[-1]

            num_ues = int(h_est.shape[4]/self.num_UE_Ant)
            PMI = []
            rate_for_selected_precoder = []
            precoding_matrices = []

            for ue_idx in range(num_ues):
                ue_ant_idx = np.arange(ue_idx*self.num_UE_Ant, (ue_idx+1)*self.num_UE_Ant)
                codebook = self.cal_codebook_type_I(tf.gather(h_est, ue_ant_idx, axis=4))
                PMI_temp, rate_for_selected_precoder_temp, precoding_matrices_temp = self.cal_PMI_type_I(codebook, tf.gather(h_est, ue_ant_idx, axis=4))
                PMI.append(PMI_temp)
                rate_for_selected_precoder.append(rate_for_selected_precoder_temp)
                precoding_matrices.append(precoding_matrices_temp)
            
            PMI = np.asarray(PMI)
            rate_for_selected_precoder = np.asarray(rate_for_selected_precoder)
            precoding_matrices = np.asarray(precoding_matrices)
            
            PMI_feedback_report = [PMI, rate_for_selected_precoder, precoding_matrices]
        
        elif self.method == 'RVQ':
            PMI_feedback_report  = self.VectorQuantizationLoader(h_est)
        else:
            raise Exception(f"The {self.method} CSI feedback mechanism has not been implemented. The simulator supports 5G standard CSI feedback and RVQ CSI feedback only.")
        
        if return_codebook:
            return PMI_feedback_report, codebook
        else:
            return PMI_feedback_report
        
    def type_II_precoder(self, curr_h_est, V, Ns, L=None, N1=None):

        curr_R = self.compute_tx_covariance(curr_h_est)
        # tx_ants = curr_h_est.shape[1]
        # curr_R = curr_R[:tx_ants//2,:tx_ants//2] + curr_R[tx_ants//2:,tx_ants//2:]
        
        N1 = int(self.N_1 if N1 is None else N1)
        L = int(self.L if L is None else L)

        m1_list, m2_list, col_idx_list = self.select_L_beams(curr_R, V, return_column_indices=True, L=L, N1=N1)

        # W_1_tmp = tf.gather(V, col_idx_list, axis=1) # [N_tx, L]
        # zeros = tf.zeros_like(W_1_tmp)
        # top = tf.concat([W_1_tmp, zeros], axis=1)
        # bottom = tf.concat([zeros, W_1_tmp], axis=1)
        # W_1 = tf.concat([top, bottom], axis=0) # [2*N1*N2, 2L]

        W_1 = tf.gather(V, col_idx_list, axis=1)  # [Nt, L]
            
        # H_A[k] = H[k] @ W_1  => [N_rx, L, N_SB]
        curr_h_est_reshaped = tf.transpose(curr_h_est, [2, 0, 1])  # [N_RB, N_rx, N_tx]
        sb_ids = tf.range(curr_h_est_reshaped.shape[0], dtype=tf.int32) // int(self.rbs_per_subband)
        curr_h_est_sb = tf.math.segment_mean(curr_h_est_reshaped, sb_ids) # [N_sb, N_rx, N_tx]
        H_A = tf.matmul(curr_h_est_sb , W_1[tf.newaxis, ...])
        H_A = tf.transpose(H_A, [1, 2, 0])  # [N_rx, L, N_SB]

        G_sb = self.compute_unquantized_bcc_per_subband(H_A, Ns) # [N_sb, L, Ns]
        G_sb_normalized = self.normalize_bcc(G_sb)

        # Debugging assert to check stream orthogonality (before quantization)
        normalized_max_off = self.assert_eq_41(H_A, G_sb)

        # WB amplitude quantization
        P_wb, k_wb = self.wb_coeff_quantize(G_sb_normalized) # TODO: check why it's mostly just giving me 1s
        W_C1 = tf.linalg.diag(tf.transpose(P_wb, perm=[1, 0]))

        # SB amplitude quantization
        W_C2, p_l_i_t_2_all = self.sb_coeff_quantize(G_sb_normalized, P_wb, N_PSK=8) # [N_sb, L, Ns]
        W_C2 = tf.transpose(W_C2, perm=[2,1,0])  # [Ns, L, N_sb]

        # Assemble W2(t)
        W_rb = self.assemble_W(W_1, W_C1, W_C2)  # [Ns, N_tx, N_sb]
        W_rb = tf.transpose(W_rb, perm=[2,0,1])  # [N_sb, Ns, N_tx]

        # Normalize
        norm_factor = 1 / tf.math.sqrt(N1 * self.N_2 * tf.reduce_sum((P_wb[tf.newaxis, ...] * p_l_i_t_2_all)**2, axis=1))
        norm_factor = tf.cast(norm_factor, self.dtype)
        W_rb = norm_factor[..., tf.newaxis] * W_rb
        W_rb = tf.transpose(W_rb, perm=[0,2,1]) # [N_sb, N_tx, Ns]

        # Validate normalization TODO: some bug here
        # col_norm2 = tf.reduce_sum(tf.abs(W_rb)**2, axis=1)  # [N_sb, Ns]
        # tf.debugging.assert_near(
        #     col_norm2,
        #     tf.ones_like(col_norm2),
        #     atol=1e-3,
        #     message="Codeword normalization failed"
        # )

        return W_rb
    
    def type_II_precoder_CB2(
        self,
        W_1,
        curr_h_est,
        Ns,
        L=None,
        N1=None,
        return_feedback_bits=False,
        w1_beam_indices=None,
        return_components=False,
    ):

        # H_A[k] = H[k] @ W_1  => [N_rx, L, N_SB]
        curr_h_est_reshaped = tf.transpose(curr_h_est, [2, 0, 1])  # [N_RB, N_rx, N_tx]
        sb_ids = tf.range(curr_h_est_reshaped.shape[0], dtype=tf.int32) // int(self.rbs_per_subband)
        curr_h_est_sb = tf.math.segment_mean(curr_h_est_reshaped, sb_ids) # [N_sb, N_rx, N_tx]
        H_A = tf.matmul(curr_h_est_sb , W_1[tf.newaxis, ...])
        H_A = tf.transpose(H_A, [1, 2, 0])  # [N_rx, L, N_SB]

        G_sb = self.compute_unquantized_bcc_per_subband(H_A, Ns) # [N_sb, L, Ns]
        G_sb_normalized = self.normalize_bcc(G_sb)

        # Debugging assert to check stream orthogonality (before quantization)
        normalized_max_off = self.assert_eq_41(H_A, G_sb)

        # WB amplitude quantization
        P_wb, k_wb = self.wb_coeff_quantize(G_sb_normalized) # TODO: check why it's mostly just giving me 1s
        W_C1 = tf.linalg.diag(tf.transpose(P_wb, perm=[1, 0]))

        # SB amplitude quantization
        if return_feedback_bits:
            W_C2, p_l_i_t_2_all, p_l_i_t_2_idx, phi_l_i_t_idx = self.sb_coeff_quantize(
                G_sb_normalized, P_wb, N_PSK=8, return_indices=True
            ) # [N_sb, L, Ns]
        else:
            W_C2, p_l_i_t_2_all = self.sb_coeff_quantize(G_sb_normalized, P_wb, N_PSK=8) # [N_sb, L, Ns]
        W_C2 = tf.transpose(W_C2, perm=[2,1,0])  # [Ns, L, N_sb]

        # Assemble W2(t)
        W_rb = self.assemble_W(W_1, W_C1, W_C2)  # [Ns, N_tx, N_sb]
        W_rb = tf.transpose(W_rb, perm=[2,0,1])  # [N_sb, Ns, N_tx]

        # Normalize
        norm_factor = 1 / tf.math.sqrt(N1 * self.N_2 * tf.reduce_sum((P_wb[tf.newaxis, ...] * p_l_i_t_2_all)**2, axis=1))
        norm_factor = tf.cast(norm_factor, self.dtype)
        W_rb = norm_factor[..., tf.newaxis] * W_rb
        W_rb = tf.transpose(W_rb, perm=[0,2,1]) # [N_sb, N_tx, Ns]

        if return_feedback_bits:
            feedback_bits = {
                "wb_amplitude_indices": tf.identity(k_wb),
                "sb_amplitude_indices": tf.identity(p_l_i_t_2_idx),
                "sb_phase_indices": tf.identity(phi_l_i_t_idx),
                "w1_beam_indices": w1_beam_indices,
            }
            if return_components:
                component_output = {"W_1": W_1, "W_C1": W_C1, "W_C2": W_C2}
                return W_rb, feedback_bits, component_output
            return W_rb, feedback_bits

        if return_components:
            component_output = {"W_1": W_1, "W_C1": W_C1, "W_C2": W_C2}
            return W_rb, component_output

        return W_rb
    
    def block_diag(self, mats):
        # mats: list of [n_i, m_i]
        r = sum([int(M.shape[0]) for M in mats])
        c = sum([int(M.shape[1]) for M in mats])
        out = tf.zeros([r, c], dtype=mats[0].dtype)
        r0 = 0
        c0 = 0
        for M in mats:
            ri = int(M.shape[0]); ci = int(M.shape[1])
            paddings = [[r0, r - (r0 + ri)], [c0, c - (c0 + ci)]]
            out += tf.pad(M, paddings)
            r0 += ri
            c0 += ci
        return out

    
    def compute_q_per_sb(self, H_ref_sb, W_ref, H_g_sb, W_g, eps=1e-9):
        
        # H_ref_sb: [N_sb, Nr, Nt_ref]
        # W_ref:    [N_sb, Nt_ref, Ns]
        # H_g_sb:   [N_sb, Nr, Nt_g]
        # W_g:      [N_sb, Nt_g, Ns]
        # returns q: [N_sb] complex

        Y_ref = tf.matmul(H_ref_sb, W_ref)  # [N_sb, Nr, Ns]
        Y_g   = tf.matmul(H_g_sb,   W_g)    # [N_sb, Nr, Ns]

        num = tf.reduce_sum(tf.math.conj(Y_g) * Y_ref, axis=[-2, -1])       # [N_sb]
        den = tf.reduce_sum(tf.abs(Y_g)**2,           axis=[-2, -1]) + eps  # [N_sb]

        q = num / tf.cast(den, num.dtype)  # [N_sb] complex
        # q = tf.exp(1j * tf.cast(tf.math.angle(q), q.dtype))

        W_g_aligned = W_g * q[:, None, None]

        Y_g_aligned = tf.matmul(H_g_sb, W_g_aligned)
        phase_err = tf.math.angle(
            tf.reduce_sum(tf.math.conj(Y_ref) * Y_g_aligned, axis=[-2, -1])
        )

        assert tf.reduce_max(tf.abs(phase_err)) < 1e-3, "Phase alignment failed"

        return q

        
    def assert_eq_41(self, H_A, G_sb):

        Ns = tf.shape(G_sb)[-1]

        H_A = tf.transpose(H_A, perm=[2,0,1]) # [N_sb, N_rx, L]
        gram_H_A = tf.matmul(H_A, H_A, adjoint_a=True) # [N_sb, L, L]
        metric = tf.matmul(G_sb, gram_H_A, adjoint_a=True) # [N_sb, Ns, L]
        metric = tf.matmul(metric, G_sb) # [N_sb, Ns, Ns]
        metric_abs = tf.abs(metric)
        off_mask = 1.0 - tf.eye(Ns, dtype=metric_abs.dtype)
        off_vals = metric_abs * off_mask

        max_off = tf.reduce_max(off_vals, axis=[-2, -1])

        diag_vals = tf.linalg.diag_part(metric_abs)
        diag_scale = tf.squeeze(tf.reduce_min(diag_vals, axis=-1, keepdims=True))

        normalized_max_off = tf.reduce_max(max_off / diag_scale)

        assert(normalized_max_off  < 1e-3), f"Orthogonality check failed, max off-diagonal value: {normalized_max_off.numpy()}" # Refer to eq. 41 of Fu et. al., "A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks", IEEE Access, 2023

        return normalized_max_off
    
    def cal_PMI_type_I(self, codebook, h_est):
        
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

    def build_sd_beam_grid(self, N1=None, N2=None):
        """
        Reference: Lee et al., "CSI Feedback for Distributed MIMO", IEEE WCNC, 2022.

        Build the candidate spatial-domain (SD) oversampled DFT beam grid exactly
        as written in the paper (no Kronecker product).

        Returns
        -------
        V : np.ndarray, shape (N1*N2, O1*N1*O2*N2)
            Each column is a candidate SD beam v_{l1,l2}.
            Columns are ordered with l2 major, l1 minor.
        """

        if N1 == None:
            N1 = self.N_1
        if N2 == None:
            N2 = self.N_2
        
        O1, O2 = self.O_1, self.O_2

        L1 = O1 * N1
        L2 = O2 * N2

        num_beams = L1 * L2
        V = np.zeros((N1 * N2, num_beams), dtype=complex)

        idx = 0

        for l2 in range(L2):

            # u_{l2} = [1, e^{j2πl2/(O2N2)}, ..., e^{j2πl2(N2-1)/(O2N2)}]^T
            u_l2 = np.exp(
                1j * 2 * np.pi * l2 * np.arange(N2) / (O2 * N2)
            )

            for l1 in range(L1):

                # v_{l1,l2} =
                # [ u_l2,
                #   e^{j2πl1/(O1N1)} u_l2,
                #   ...
                #   e^{j2πl1(N1-1)/(O1N1)} u_l2 ]^T

                beam_blocks = []
                for n1 in range(N1):
                    phase = np.exp(1j * 2 * np.pi * l1 * n1 / (O1 * N1))
                    beam_blocks.append(phase * u_l2)

                v_l1_l2 = np.concatenate(beam_blocks, axis=0)

                # Normalize (unit norm, consistent with Type-II usage)
                v_l1_l2 /= np.linalg.norm(v_l1_l2)

                V[:, idx] = v_l1_l2
                idx += 1

        return V
    
    def compute_tx_covariance(self, h_est):
        """
        h_est: [N_rx, N_tx, N_RB]
        returns: [N_tx, N_tx]
        """

        H = tf.cast(h_est, self.dtype)   # <-- THIS fixes complex128 -> complex64

        if H.shape.rank != 3:
            raise ValueError(f"h_est must be rank-3 [N_rx,N_tx,N_RB], got {H.shape}")

        # [N_RB, N_rx, N_tx]
        Hk = tf.transpose(H, perm=[2, 0, 1])

        # Rk = Hk^H Hk : [N_RB, N_tx, N_tx]
        Rk = tf.matmul(Hk, Hk, adjoint_a=True)

        # average over RBs
        R = tf.reduce_mean(Rk, axis=0)

        return R

    def select_L_beams(self, R, V, L=None, N1=None, N2=None, return_column_indices=True):
        """
        Reference: Fu et. al., "A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks", IEEE Access, 2023, Algorithm 1 beam selection.
        Modified to correctly handle N2 = 1 (1D array): only exclude n1 per iteration.

        Inputs
        ------
        R : np.ndarray, shape (P, P) where P = N1*N2
            TX-side covariance.
        V : np.ndarray, shape (P, L1*L2)
            Beam grid from build_sd_beam_grid().
        L : int or None
            Number of beams to select. If None, uses self.L.
        """
        
        N1 = int(self.N_1 if N1 is None else N1)
        N2 = int(self.N_2 if N2 is None else N2)
        
        O1, O2 = int(self.O_1), int(self.O_2)
        
        L = int(self.L if L is None else L)

        L1 = O1 * N1
        L2 = O2 * N2
        P  = N1 * N2

        if V.shape != (P, L1 * L2):
            raise ValueError(f"V has shape {V.shape}, expected ({P}, {L1*L2}).")

        if R.shape != (P, P):
            raise ValueError(f"R must be ({P},{P}); got {R.shape}.")

        # --- feasibility checks ---
        if N2 == 1:
            # only N1 distinct orthogonal beams available (for a fixed q1)
            if L > N1:
                raise ValueError(f"For N2=1, must have L <= N1. Got L={L}, N1={N1}.")
        else:
            # Fu Algorithm 1 removes one n1 and one n2 per iteration => L <= min(N1,N2)
            if L > min(N1, N2):
                raise ValueError(f"L={L} too large; must satisfy L <= min(N1,N2)={min(N1,N2)}.")

        # column index mapping: l2-major, l1-minor
        def col_from_m(m1, m2):
            return int(m2) * L1 + int(m1)

        def metric_for_col(col_idx):
            v = V[:, col_idx]
            v = v[:, np.newaxis]  # make column vector
            return float(np.real(np.vdot(v, R @ v)))

        # ---- i = 0: search over all (m1,m2) ----
        best_val = -np.inf
        best_m1 = 0
        best_m2 = 0
        for m2 in range(L2):
            base = m2 * L1
            for m1 in range(L1):
                col = base + m1
                val = metric_for_col(col)
                if val > best_val:
                    best_val = val
                    best_m1 = m1
                    best_m2 = m2

        # offsets from first selected beam
        q1 = best_m1 % O1
        q2 = best_m2 % O2

        # coarse indices
        n1_0 = best_m1 // O1
        n2_0 = best_m2 // O2  # will be 0 when N2=1

        # remaining candidate sets
        N1_set = set(range(N1))
        N1_set.discard(int(n1_0))

        if N2 > 1:
            N2_set = set(range(N2))
            N2_set.discard(n2_0)
        else:
            N2_set = {0}  # fixed

        m1_list = [best_m1]
        m2_list = [best_m2]
        col_idx_list = [col_from_m(best_m1, best_m2)]

        # ---- i = 1..L-1 ----
        for _ in range(1, L):

            best_val = -np.inf
            best_n1 = None
            best_n2 = None

            # For N2=1 this loops once with n2=0
            for n2 in N2_set:
                m2 = O2 * n2 + q2
                for n1 in N1_set:
                    m1 = O1 * n1 + q1
                    col = col_from_m(m1, m2)
                    val = metric_for_col(col)
                    if val > best_val:
                        best_val = val
                        best_n1 = n1
                        best_n2 = n2

            # commit selection
            m1_i = O1 * best_n1 + q1
            m2_i = O2 * best_n2 + q2

            m1_list.append(int(m1_i))
            m2_list.append(int(m2_i))
            col_idx_list.append(col_from_m(m1_i, m2_i))

            # remove used indices
            N1_set.discard(best_n1)
            if N2 > 1:
                N2_set.discard(best_n2)  # only in true 2D case

        if return_column_indices:
            return m1_list, m2_list, col_idx_list
        return m1_list, m2_list
    
    def compute_unquantized_bcc_per_subband(self, H_A, Ns):
        """
        Reference: Fu et. al., "A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks", IEEE Access, 2023
        
        Compute unquantized Type-II BCC coefficients per subband.

        Parameters
        ----------
        H_A : tf.Tensor, shape [N_rx, L, N_sb]
            Projected channel H_A[k] = H[k] B.

        Returns
        -------
        G_sb : tf.Tensor, shape [N_sb, L, Ns]
            Unquantized beam-domain coefficient matrix per subband.
            Each subband gets its own top-Ns eigenvectors of R_A^(s).
        """
        H_A = tf.cast(H_A, self.dtype)

        # [N_sb, N_rx, L]
        Hk = tf.transpose(H_A, [2, 0, 1])

        # Per-RB beam-domain gram matrix: G = Hk^H Hk  -> [N_sb, L, L]
        G = tf.matmul(Hk, Hk, adjoint_a=True)

        # Eigen-decomposition per subband (Hermitian): returns ascending evals
        evals, evecs = tf.linalg.eigh(G)        # evecs: [N_sb, L, L]
        idx = tf.argsort(tf.abs(evals), axis=-1, direction="DESCENDING")[:, :Ns]
        G_sb = tf.gather(
            evecs, idx, batch_dims=1, axis=-1
        )  # [N_sb, L, Ns] largest Ns eigenvectors per subband

        return G_sb
    
    def normalize_bcc(self, G_sb):
        """
        Reference: Fu et. al., "A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks", IEEE Access, 2023
        BCC normalization (Eq. 42–45).

        G_sb : [N_sb, L, Ns]

        Returns
        -------
        G_sb_norm : [N_sb, L, Ns]
        """
        
        G_sb = tf.cast(G_sb, self.dtype)

        # Power summed over subbands
        power = tf.reduce_sum(tf.abs(G_sb)**2, axis=0)  # [L, Ns]

        # Argmax over beam index
        i_prime = tf.argmax(power, axis=0)              # [Ns]

        N_sb = tf.shape(G_sb)[0]
        Ns   = tf.shape(G_sb)[2]

        # Gather reference coefficients w_{2,i',t}^{(l)}
        ref = tf.stack([
            G_sb[:, i_prime[l], l] for l in range(Ns)
        ], axis=-1)                                     # [N_sb, Ns]

        # Normalize column-wise
        G_sb_norm = G_sb / (ref[:, tf.newaxis, :])

        return G_sb_norm
    
    def get_typeII_wb_amplitude_levels_P1(self):
        """
        TS 38.214 Type-II WB amplitude levels P1 (k^(1)=0..7).
        Returns real tensor of shape [8].
        """
        P1 = [0.0,
            1.0/64.0,
            1.0/32.0,
            1.0/16.0,
            1.0/8.0,
            1.0/4.0,
            1.0/2.0,
            1.0]
        return tf.constant(P1, dtype=self.real_dtype)


    def wb_coeff_quantize(self, W2_tilde_sb):
        """
        Reference: Fu et. al., "A Tutorial on Downlink Precoder Selection Strategies for 3GPP MIMO Codebooks", IEEE Access, 2023
        Step 4 (Eq. 46): wideband coefficient calculation and quantization,
        using TS 38.214 WB amplitude codebook P1.

        Inputs
        ------
        W2_tilde_sb : tf.Tetf.abs(tf.abs(D_l) - P2[tf.newaxis, tf.newaxis, :])nsor, shape [N_sb, L, Ns]
            Normalized BCC matrix (after Fu Eqs. 42-45), per subband.
        i_prime : tf.Tensor or None, shape [Ns]
            Strongest-beam index per layer (Fu Eq. 43). If provided, we force
            the strongest-beam WB amplitude to 1 (k^(1)=7) explicitly.

        Outputs
        -------
        P_wb : tf.Tensor, shape [L, Ns]
            Quantized WB amplitudes p_{l,i}^{(1)}.
        k_wb : tf.Tensor, shape [L, Ns], dtype int32
            WB amplitude indices k^{(1)} in {0..7}.
        """
        W2_tilde_sb = tf.cast(W2_tilde_sb, self.dtype)          # [N_sb, L, Ns]
        P1 = self.get_typeII_wb_amplitude_levels_P1()          # [8] real

        # max_t |w_{2,i,t}^{(l)}| over subbands t
        mag_max = tf.reduce_max(tf.abs(W2_tilde_sb), axis=0)   # [L, Ns], real

        # Nearest-neighbor quantization to P1:
        # dist: [L, Ns, 8]
        dist = tf.abs(mag_max[..., tf.newaxis] - P1[tf.newaxis, tf.newaxis, :])
        k_wb = tf.argmin(dist, axis=-1, output_type=tf.int32)  # [L, Ns]
        P_wb = tf.gather(P1, k_wb)                             # [L, Ns] real

        return P_wb, k_wb


    def get_typeII_sb_amplitude_levels_P2(self):
        # TS 38.214 Table 5.2.2.2.3-3: {sqrt(1/2), 1}
        return tf.convert_to_tensor([tf.sqrt(0.5), 1.0], dtype=self.real_dtype)

    def sb_coeff_quantize(self, G_sb_normalized, P_wb, N_PSK=8, eps=1e-12, return_indices=False):
        """
        Subband amplitude quantization using P2={1/2,1}.

        Inputs
        ------
        G_sb_normalized : tf.Tensor [N_sb, L, Ns] complex
            Normalized BCC coefficients (after Fu Eq. 42-45 style normalization)
        P_wb : tf.Tensor [L, Ns] real
            WB amplitudes from Step 4

        Returns
        -------
        P_sb : tf.Tensor [N_sb, L, Ns] real
            Quantized SB amplitudes
        k_sb : tf.Tensor [N_sb, L, Ns] int32
            Indices in {0,1} mapping to {1/2,1}
        """
        G = tf.cast(G_sb_normalized, self.dtype)  # [N_sb, L, Ns]
        P_wb = tf.cast(P_wb, self.dtype)

        P2 = self.get_typeII_sb_amplitude_levels_P2()  # [2]

        ph = tf.constant([2*np.pi*c/N_PSK for c in range(N_PSK)],
                        dtype=self.real_dtype)          # [N_PSK]

        # calculate quantized subband amplitudes and phases
        W_C2_l = []
        p_l_i_t_2_all = []
        p_l_i_t_2_idx_all = []
        phi_l_i_t_idx_all = []
        for l in range(G_sb_normalized.shape[-1]):
            W_C1_l_inv = tf.where(tf.abs(P_wb[:,l]) > 0, 1.0/P_wb[:,l], tf.zeros_like(P_wb[:,l]))
            W_C1_l_inv = tf.linalg.diag(W_C1_l_inv)
            W_2_l = tf.gather(G_sb_normalized, l, axis=-1)
            D_l = tf.matmul(W_C1_l_inv[tf.newaxis, ...], W_2_l[..., tf.newaxis])

            abs_dist = tf.abs(tf.abs(D_l) - P2[tf.newaxis, tf.newaxis, :])
            p_l_i_t_2_idx = tf.argmin(abs_dist, axis=-1, output_type=tf.int32)                             # [N_sb,L]
            p_l_i_t_2 = tf.gather(P2, p_l_i_t_2_idx)
            p_l_i_t_2_all.append(p_l_i_t_2[..., tf.newaxis])
            p_l_i_t_2_idx_all.append(p_l_i_t_2_idx[..., tf.newaxis])

            theta_0_2pi = tf.math.mod(tf.math.angle(D_l) + 2*np.pi, 2*np.pi)
            phi_dist = tf.abs(theta_0_2pi - ph[tf.newaxis, tf.newaxis, :])
            phi_l_i_t_idx = tf.argmin(phi_dist, axis=-1, output_type=tf.int32)                             # [N_sb,L]
            phi_l_i_t = tf.gather(ph, phi_l_i_t_idx)
            phi_l_i_t_idx_all.append(phi_l_i_t_idx[..., tf.newaxis])

            p_phi = tf.cast(p_l_i_t_2, self.dtype) * tf.exp(
                1j * tf.cast(phi_l_i_t, self.dtype)
            )

            W_C2_l.append(p_phi[..., tf.newaxis])

        W_C2_l = tf.concat(W_C2_l, axis=-1)
        p_l_i_t_2_all = tf.concat(p_l_i_t_2_all, axis=-1)
        p_l_i_t_2_idx_all = tf.concat(p_l_i_t_2_idx_all, axis=-1)
        phi_l_i_t_idx_all = tf.concat(phi_l_i_t_idx_all, axis=-1)

        if return_indices:
            return W_C2_l, p_l_i_t_2_all, p_l_i_t_2_idx_all, phi_l_i_t_idx_all

        return W_C2_l, p_l_i_t_2_all 
    
    def phase_quantize_psk(self, G_sb_normalized, N_PSK=4):
        """
        Phase quantization to N_PSK-PSK (N_PSK typically 4 or 8).

        Returns
        -------
        E_phi : tf.Tensor [N_sb, L, Ns] complex
            exp(j * quantized_phase)
        c_idx : tf.Tensor [N_sb, L, Ns] int32
            phase indices in {0,...,N_PSK-1}
        """
        if N_PSK not in (4, 8):
            raise ValueError(f"N_PSK must be 4 or 8; got {N_PSK}")

        G = tf.cast(G_sb_normalized, self.dtype)
        ang = tf.math.angle(G)                                  # [-pi,pi]
        ang = tf.where(ang < 0.0, ang + 2*np.pi, ang)          # [0,2pi)

        ph = tf.constant([2*np.pi*c/N_PSK for c in range(N_PSK)],
                        dtype=self.real_dtype)          # [N_PSK]

        # circular distance to each PSK angle
        diff = ang[..., tf.newaxis] - ph[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
        diff = tf.math.floormod(diff + np.pi, 2*np.pi) - np.pi
        c_idx = tf.argmin(tf.abs(diff), axis=-1, output_type=tf.int32)  # [N_sb,L,Ns]
        c_idx = tf.squeeze(c_idx, axis=0)

        ph_q = tf.gather(ph, c_idx)                              # [N_sb,L,Ns]
        E_phi = tf.exp(1j * tf.cast(ph_q, self.dtype)) # [N_sb,L,Ns] complex
        E_phi = tf.cast(E_phi, self.dtype)

        return E_phi, c_idx

    def assemble_W(self, W_1, W_C1, W_C2):
        """
        Assemble quantized W2 per subband.

        W_1 : [L, Ns] real
        W_C1 : [L, Ns] real
        W_C2 : [Ns, L, N_sb] complex
        E_phi: [N_sb, L, Ns] complex

        Returns
        -------
        W2_q : [N_sb, L, Ns] complex
        """
        W_1 = tf.cast(W_1, self.dtype)
        W_1 = W_1[tf.newaxis, :, :]                     # [1,L,L]
        W_C1 = tf.cast(W_C1, self.dtype)           # [Ns, L, L]
        W_C2 = tf.cast(W_C2, self.dtype)                # [Ns,L,N_sb]

        W = tf.matmul(W_1, W_C1)        # [1,L,L] @ [Ns, L, L] -> [Ns, L, L]
        W = tf.matmul(W, W_C2)        # [Ns,L,L] @ [Ns,L,N_sb] -> [Ns,L,N_sb]

        return W


    def cal_codebook_type_I(self, h_est):
        """
        Computes Type I PMI codebook for 2/4 tx antennas
        Consult 3GPP TS 38.214 Section 5 for details
        """

        N_t = h_est.shape[4]
        P_CSI_RS = N_t

        if N_t == 2:

            if self.num_tx_streams == 1:
                # Table 5.2.2.2.1-1, v=1, codebook indices 0..3
                # W = (1/sqrt(2)) * [1; exp(j*pi*n/2)], n=0..3
                n_all = np.arange(0, 4)

                W = np.zeros((len(n_all), N_t, self.num_tx_streams), dtype=complex)

                for n in n_all:
                    phi_n = np.exp(1j * np.pi * n / 2)  # 1, j, -1, -j
                    W[n, :, 0] = np.array([1.0, phi_n], dtype=complex)

                # normalize
                W = 1 / np.sqrt(2) * W

            elif self.num_tx_streams == 2:
                # Table 5.2.2.2.1-1, v=2, codebook indices 0..1
                # index 0: (1/2) * [[1,  1],
                #                  [1, -1]]
                # index 1: (1/2) * [[1,  1],
                #                  [j, -j]]
                i_cb = np.arange(0, 2)

                W = np.zeros((len(i_cb), N_t, self.num_tx_streams), dtype=complex)

                W[0, :, :] = np.array([[1,  1],
                                    [1, -1]], dtype=complex)

                W[1, :, :] = np.array([[1,   1],
                                    [1j, -1j]], dtype=complex)

                # normalize
                W = 1 / 2 * W

            else:
                raise Exception(
                    f"2-port Type I Single-Panel codebook supports 1 or 2 layers only; got {self.num_tx_streams}."
                )
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

    def quantize_feedback(self, h_freq_csi, cfg, rg_csi, *, donald_hack=True, quantization_debug=False):
        """Quantize CSI feedback using the internal random vector codebook."""

        num_tx_ant = h_freq_csi.shape[4]
        h_freq_per_rx = []

        csi_effective_subcarriers = rg_csi.num_effective_subcarriers
        csi_guard_carriers_1 = rg_csi.num_guard_carriers[0]
        csi_guard_carriers_2 = rg_csi.num_guard_carriers[1]
        effective_subcarriers = (csi_effective_subcarriers // cfg.num_tx_streams) * cfg.num_tx_streams
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += csi_guard_carriers_1
        guard_carriers_2 += csi_guard_carriers_2
        num_tx_ant = h_freq_csi.shape[4]
        for i_rxnode in range(cfg.num_tx_streams):
            if guard_carriers_2 == 0:
                h_freq_rx = h_freq_csi[:, :, i_rxnode*2:(i_rxnode+1)*2, : , :, :, guard_carriers_1:]
            else:
                h_freq_rx = h_freq_csi[:, :, i_rxnode*2:(i_rxnode+1)*2, : , :, :, guard_carriers_1:-guard_carriers_2]

            H = tf.transpose(h_freq_rx, perm=[0, 1, 3, 5, 6, 2, 4])

            num_syms = H.shape[3]
            H = tf.reduce_mean(H, axis=3, keepdims=True)
            n_sc = H.shape[4]; B = H.shape[0]
            num_rbs = n_sc // cfg.rb_size

            if n_sc % cfg.rb_size == 0:
                H = tf.reshape(H, [B , 1, 1, 1, n_sc//cfg.rb_size, cfg.rb_size, 2, num_tx_ant])
                num_residual_subcarriers = 0
                H = tf.reduce_mean(H, axis=5, keepdims=True)
                n_sc_less_residual = n_sc
            else:
                num_residual_subcarriers = n_sc % cfg.rb_size
                n_sc_less_residual = n_sc - (num_residual_subcarriers)
                H_less_last_rb = H[:, :, :, :, :-(cfg.rb_size + num_residual_subcarriers)]
                H_last_rb = H[:, :, :, :, -(cfg.rb_size + num_residual_subcarriers):]
                H_less_last_rb = tf.reshape(H_less_last_rb, [B , 1, 1, 1, num_rbs - 1, cfg.rb_size, 2, num_tx_ant])
                H_last_rb = tf.reshape(H_last_rb, [B , 1, 1, 1, 1, cfg.rb_size + num_residual_subcarriers, 2, num_tx_ant])
                H_less_last_rb = tf.reduce_mean(H_less_last_rb, axis=5, keepdims=True)
                H_last_rb = tf.reduce_mean(H_last_rb, axis=5, keepdims=True)
                H = tf.concat([H_less_last_rb, H_last_rb], axis=4)

            if donald_hack and ("DIRECT" not in cfg.precoding_method):
                H_avg = tf.reduce_mean(H, axis=-2)
                H_avg_norm = (tf.linalg.norm(H_avg, axis=-1, keepdims=True) + 1e-12)
                H_avg_normalized = H_avg / H_avg_norm
                H_avg_normalized_quantized = self(H_avg_normalized)
                H_avg_normalized_reconstructed = self(H_avg_normalized_quantized)
                H_avg_reconstructed = (H_avg_normalized_reconstructed)
                H_avg_reconstructed = tf.tile(H_avg_reconstructed, [1, 1, 1, num_syms, 1, cfg.rb_size, 1])
                H_avg_reconstructed = tf.reshape(H_avg_reconstructed, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])
                if num_residual_subcarriers != 0:
                    H_avg_reconstructed = tf.concat([
                        H_avg_reconstructed,
                        H_avg_reconstructed[:, :, -(num_residual_subcarriers):, :, :],
                    ], axis=2)
                if quantization_debug:
                    H_avg1 = tf.tile(H_avg, [1, 1, 1, num_syms, 1, cfg.rb_size, 1])
                    H_avg1 = tf.reshape(H_avg1, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])
                    print(f"For RX node {i_rxnode}, quantization distortion (Frobenius norm) norm(actual - reconstructed) /norm(actual):" +
                        f" {tf.linalg.norm(H_avg1 - H_avg_reconstructed) / tf.linalg.norm(H_avg1)}")
                h_freq_per_rx.append(tf.transpose(H_avg_reconstructed, perm=[0, 3, 4, 1, 2]))
            else:
                s, u , v = tf.linalg.svd(H)
                v_largest = v[..., 0]
                v_largest = tf.tile(v_largest, [1, 1, 1, num_syms, 1, cfg.rb_size, 1])
                v_largest = tf.reshape(v_largest, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])
                if num_residual_subcarriers != 0:
                    v_largest = tf.concat([
                        v_largest,
                        v_largest[:, :, -(num_residual_subcarriers):, :, :],
                    ], axis=2)
                v_largest_quantized = self(v_largest)
                v_largest_reconstructed = self(v_largest_quantized)
                vh = tf.linalg.adjoint(v_largest_reconstructed)
                h_freq_per_rx.append(tf.transpose(vh, perm=[0, 4, 3, 1, 2]))

        h_freq_quantized = tf.concat(h_freq_per_rx, axis=1)
        first_subcarrier = tf.repeat(h_freq_quantized[..., 0:1], repeats=guard_carriers_1, axis=-1)
        last_subcarrier = tf.repeat(h_freq_quantized[..., -1:], repeats=guard_carriers_2, axis=-1)
        return tf.concat([first_subcarrier, h_freq_quantized, last_subcarrier], axis=-1)


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