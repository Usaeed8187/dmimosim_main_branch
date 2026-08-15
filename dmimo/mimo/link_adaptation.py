import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna

from dmimo.mimo import rankAdaptation


# 3GPP TS 38.214, Table 5.1.3.1-1 (PDSCH MCS table 1).  Target
# code rates are specified by 3GPP as R x 1024; keep the integer values here
# and divide only when constructing the array used by the PHY.
MCS_TABLE_38_214_QM_RATE_X1024 = np.array(
    [
        (2, 120), (2, 157), (2, 193), (2, 251), (2, 308),
        (2, 379), (2, 449), (2, 526), (2, 602), (2, 679),
        (4, 340), (4, 378), (4, 434), (4, 490), (4, 553),
        (4, 616), (4, 658),
        (6, 438), (6, 466), (6, 517), (6, 567), (6, 616),
        (6, 666), (6, 719), (6, 772), (6, 822), (6, 873),
        (6, 910), (6, 948),
    ],
    dtype=np.float64,
)


_LONG_BETA = np.array(
    [1.49, 1.61, 3.36, 4.56, 6.42, 13.76, 25.16, 28.38],
    dtype=np.float64,
)
_LONG_SINR_DB = np.array(
    [0.2, 4.3, 5.9, 8.1, 10.3, 14.1, 18.7, 21.0],
    dtype=np.float64,
)
_LONG_MCS_CANDIDATES = np.array(
    [
        (2, 0.30), (2, 0.60),
        (4, 0.37), (4, 0.50), (4, 0.60), (4, 0.66),
        (6, 0.55), (6, 0.75), (6, 0.85),
    ],
    dtype=np.float64,
)


def _linear_interp_with_extrapolation(x, xp, fp):
    """One-dimensional interpolation with linear end extrapolation."""
    x = np.asarray(x, dtype=np.float64)
    values = np.interp(x, xp, fp)
    below = x < xp[0]
    above = x > xp[-1]
    values[below] = fp[0] + (x[below] - xp[0]) * (
        (fp[1] - fp[0]) / (xp[1] - xp[0])
    )
    values[above] = fp[-1] + (x[above] - xp[-1]) * (
        (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    )
    return values


def get_link_adaptation_table(table_name):
    """Return EESM parameters and ``[Qm, R]`` candidates.

    The modulation orders and target rates in the ``38.214`` table are the
    exact entries of PDSCH MCS Table 1.  3GPP does not standardize EESM beta
    values or effective-SINR switching thresholds.  Until link-level BLER
    calibration is available for every MCS, those parameters are interpolated
    over spectral efficiency from the simulator's legacy ``long`` anchors.
    """
    table_name = str(table_name).lower()
    if table_name == "long":
        return _LONG_BETA.copy(), _LONG_SINR_DB.copy(), _LONG_MCS_CANDIDATES.copy()
    if table_name == "short":
        return (
            np.array([1.61, 6.42, 28.38], dtype=np.float64),
            np.array([4.3, 10.3, 22.7], dtype=np.float64),
            np.array([(2, 0.60), (4, 0.50), (6, 0.65)], dtype=np.float64),
        )
    if table_name != "38.214":
        raise ValueError(
            f"Unknown link-adaptation table '{table_name}'. "
            "Expected 'short', 'long', or '38.214'."
        )

    candidates = MCS_TABLE_38_214_QM_RATE_X1024.copy()
    candidates[:, 1] /= 1024.0
    spectral_efficiency = candidates[:, 0] * candidates[:, 1]

    # The legacy long table has eight calibrated EESM anchors and a ninth
    # terminal candidate.  Interpolate from the eight parameterized anchors.
    anchor_spectral_efficiency = (
        _LONG_MCS_CANDIDATES[:-1, 0] * _LONG_MCS_CANDIDATES[:-1, 1]
    )
    beta = _linear_interp_with_extrapolation(
        spectral_efficiency, anchor_spectral_efficiency, _LONG_BETA
    )
    sinr_db = _linear_interp_with_extrapolation(
        spectral_efficiency, anchor_spectral_efficiency, _LONG_SINR_DB
    )
    return beta, sinr_db, candidates


def sionna_ldpc5g_supported_mcs_mask(mcs_candidates, codeword_length):
    """Return entries supported by Sionna's unsegmented LDPC5GEncoder.

    This mirrors the encoder's base-graph and code-length checks.  It is a
    limitation of the current simulator's one-LDPC-block-per-codeword layout,
    not a restriction of the 38.214 MCS table itself.
    """
    candidates = np.asarray(mcs_candidates, dtype=np.float64)
    n = int(codeword_length)
    supported = np.zeros(candidates.shape[0], dtype=bool)
    for idx, rate_target in enumerate(candidates[:, 1]):
        k = int(n * rate_target)
        if k < 12 or k > 8448 or n < 0 or n > 316 * 384:
            continue
        rate = k / n
        if rate < 1 / 5 or rate > 0.95:
            continue
        if k <= 292:
            base_graph = "bg2"
        elif k <= 3824 and rate <= 0.67:
            base_graph = "bg2"
        elif rate <= 0.25:
            base_graph = "bg2"
        else:
            base_graph = "bg1"
        if base_graph == "bg1":
            supported[idx] = k <= 8448 and rate >= 1 / 3
        else:
            supported[idx] = k <= 3840 and rate >= 1 / 5
    return supported


def project_mcs_indices_to_sionna_supported(
    mcs_indices,
    mcs_candidates,
    codeword_length,
):
    """Map recommendations to the closest no-more-aggressive supported MCS."""
    candidates = np.asarray(mcs_candidates, dtype=np.float64)
    supported = sionna_ldpc5g_supported_mcs_mask(candidates, codeword_length)
    if not np.any(supported):
        raise ValueError(
            f"No MCS is supported for LDPC codeword length {codeword_length}."
        )
    spectral_efficiency = candidates[:, 0] * candidates[:, 1]
    supported_indices = np.flatnonzero(supported)
    projected = np.empty(np.asarray(mcs_indices).shape, dtype=np.int64)
    for output_index, recommended in np.ndenumerate(np.asarray(mcs_indices)):
        recommended = int(np.clip(recommended, 0, candidates.shape[0] - 1))
        eligible = supported_indices[
            spectral_efficiency[supported_indices]
            <= spectral_efficiency[recommended] + 1e-12
        ]
        if eligible.size == 0:
            selected = supported_indices[np.argmin(spectral_efficiency[supported_indices])]
        else:
            selected = eligible[np.argmax(spectral_efficiency[eligible])]
        projected[output_index] = int(selected)
    return projected

class linkAdaptation(Layer):
    """link adaptation for SU-MIMO and MU-MIMO"""

    def __init__(self,
                num_bs_ant,
                num_ue_ant,
                architecture,
                sinrdb,
                nfft,
                N_s,
                data_sym_position,
                lookup_table_size="38.214",
                dtype=tf.complex64,
                **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        
        self.num_BS_Ant = num_bs_ant
        self.num_UE_Ant = num_ue_ant
        self.nfft = nfft
        self.architecture = architecture
        if self.architecture == 'SU-MIMO':
            sinr_linear = 10**(sinrdb/10)
            sinr_linear = np.sum(sinr_linear, axis=(2))
            self.sinr_linear = np.mean(sinr_linear)
            precoder= 'SVD'
        elif self.architecture == 'MU-MIMO':
            sinr_linear = 10**(sinrdb/10)
            self.sinr_linear = sinr_linear
            precoder= 'BD'
        else:
            raise Exception(f"Rank adaptation for {self.architecture} has not been implemented.")

        self.data_sym_position = data_sym_position
        self.num_data_symbols = self.data_sym_position.shape[0]

        self.use_mmse_eesm_method = True
        self.lookup_table_size = lookup_table_size
        # Sionna's LDPC5GEncoder supports code rates >= 1/5.  Table-1 MCS
        # indices 0--2 remain represented by get_link_adaptation_table(), but
        # require additional repetition/rate matching before this PHY can use
        # them for an actual transmission.
        self.minimum_mcs_index = 3 if str(lookup_table_size).lower() == "38.214" else 0

        self.N_s = N_s
        
        self.rank_adaptation = rankAdaptation(num_bs_ant, num_ue_ant, architecture, sinrdb, nfft, precoder=precoder)


    def call(self, h_est, channel_type, return_mcs_index=False):

        if self.architecture == "SU-MIMO":
            feedback_report  = self.generate_link_SU_MIMO(h_est, channel_type)
        elif self.architecture == "MU-MIMO":
            feedback_report = self.generate_link_MU_MIMO(h_est, channel_type, return_mcs_index)
        
        return feedback_report

    def generate_link_SU_MIMO(self, h_est, channel_type):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        total_num_symbols = h_est.shape[5]

        H_freq = tf.squeeze(h_est)
        H_freq = tf.transpose(H_freq, perm=[3,0,1,2])

        if self.use_mmse_eesm_method:

            beta_list, refer_sinr_db, mcs_candidates = get_link_adaptation_table(
                self.lookup_table_size
            )


            qam_order_arr = np.zeros((self.N_s))
            code_rate_arr = np.zeros((self.N_s))
            cqi_snr = np.zeros((self.N_s))

            if self.N_s == 1:
                
                avg_sinr = self.sinr_linear

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
                n_var = self.rank_adaptation.cal_n_var(h_eff, self.sinr_linear)
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
    
    def generate_link_MU_MIMO(self, h_est, channel_type, return_mcs_index):

        N_t = h_est.shape[4]
        N_r = h_est.shape[2]
        num_rx_nodes = int(N_r/self.num_UE_Ant)

        h_est = h_est[0:1,...]
        H_freq = tf.squeeze(h_est)
        H_freq = tf.transpose(H_freq, perm=[3,0,1,2])

        mcs_index = None

        if self.use_mmse_eesm_method:

            beta_list, refer_sinr_db, mcs_candidates = get_link_adaptation_table(
                self.lookup_table_size
            )


            qam_order_arr = np.zeros((self.N_s, num_rx_nodes))
            code_rate_arr = np.zeros((self.N_s, num_rx_nodes))
            cqi_snr = np.zeros((self.N_s, num_rx_nodes))
            mcs_indices = np.zeros((self.N_s, num_rx_nodes))

            if self.N_s == 1:
                
                for rx_node_idx in range(num_rx_nodes):

                    curr_sinr_linear = self.sinr_linear[:, :, rx_node_idx, :]

                    sinr_eff_list = []
                    for beta in beta_list:
                        sinr_eff = -beta * np.log(np.mean(np.exp(-curr_sinr_linear / beta)))
                        sinr_eff_dB = 10*np.log10(sinr_eff)
                        sinr_eff_list.append(sinr_eff_dB)

                    curr_qam_order, curr_code_rate, cqi_snr_tmp, mcs_index = self.lookup_table(sinr_eff_list, refer_sinr_db, mcs_candidates, return_mcs_index)

                    qam_order_arr[0, rx_node_idx] = curr_qam_order
                    code_rate_arr[0, rx_node_idx] = curr_code_rate
                    cqi_snr[0, rx_node_idx] = cqi_snr_tmp
                    mcs_indices[0, rx_node_idx] = mcs_index
                
            else:

                h_eff = self.rank_adaptation.calculate_effective_channel(self.N_s, h_est)
                
                for rx_node_idx in range(num_rx_nodes):

                    if rx_node_idx == 0:
                        ant_indices = np.arange(self.num_BS_Ant)
                    else:
                        ant_indices = np.arange((rx_node_idx-1)*self.num_UE_Ant  + self.num_BS_Ant, rx_node_idx*self.num_UE_Ant + self.num_BS_Ant)
                    curr_sinr_linear = np.sum(self.sinr_linear[ant_indices])

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

            return [qam_order_arr, code_rate_arr, cqi_snr, mcs_indices]
        else:
            raise Exception(f"The non-EESM methods have not been implemented.")


        
    
    def lookup_table(
        self,
        sinr_eff_list,
        refer_sinr_db,
        mcs_candidates,
        return_mcs_index=False,
    ):

        assert len(sinr_eff_list) == refer_sinr_db.shape[0]

        # The 38.214 table has one calibrated/interpolated EESM threshold per
        # candidate.  Legacy tables retain their boundary-threshold behavior
        # for backward compatibility.
        if refer_sinr_db.shape[0] == mcs_candidates.shape[0]:
            minimum_mcs_index = min(
                int(getattr(self, "minimum_mcs_index", 0)),
                mcs_candidates.shape[0] - 1,
            )
            admissible = np.flatnonzero(
                np.asarray(sinr_eff_list) >= np.asarray(refer_sinr_db)
            )
            admissible = admissible[admissible >= minimum_mcs_index]
            mcs_idx = (
                int(admissible[-1])
                if admissible.size
                else minimum_mcs_index
            )
            curr_qam_order, curr_code_rate = mcs_candidates[mcs_idx, :]
            cqi_snr = refer_sinr_db[mcs_idx]
            return curr_qam_order, curr_code_rate, cqi_snr, (
                mcs_idx if return_mcs_index else None
            )

        mcs_idx = 0
        for idx in range(refer_sinr_db.shape[0]):
            if sinr_eff_list[idx] > refer_sinr_db[idx]:
                mcs_idx += 1
        
        # mcs_idx = np.max([mcs_idx-1, 0])

        [curr_qam_order, curr_code_rate] = mcs_candidates[mcs_idx, :]

        if not return_mcs_index:
            mcs_index = None

        cqi_snr = refer_sinr_db[min(mcs_idx, refer_sinr_db.shape[0] - 1)]
        return curr_qam_order, curr_code_rate, cqi_snr, mcs_idx
