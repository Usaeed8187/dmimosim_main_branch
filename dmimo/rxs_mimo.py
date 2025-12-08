import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.ofdm import LMMSEEqualizer, MMSEPICDetector
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, RandomInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber, compute_bler
from sionna.channel import AWGN
from sionna.utils import flatten_last_dims

from dmimo.config import SimConfig, Ns3Config
from dmimo.channel import lmmse_channel_estimation
from dmimo.mimo import quantized_CSI_feedback, phase_3_mu_mimo_uplink_precoder, phase_3_sic_lmmse_decoder

class RxSquad(Model):
    """
    Implement Rx Squad data transmission in phase 3 (P3)

    For the Rx squad uplink, OFDMA is required for more than 2 UEs. Here we simulate
    the average performance for each pair of UEs uplink across all subcarriers, i.e.,
    two UEs are transmitting simultaneously to the BS with 4x4 MIMO channels.

    """

    def __init__(self, cfg: SimConfig, ns3cfg: Ns3Config, rxs_bits_per_frame: int, rxs_chans, coderate=2/3, **kwargs):
        """
        Initialize RxSquad simulation

        :param cfg: simulation settings
        :param rxs_bits_per_frame: total number of bits per subframe/slot for all UEs
        """
        super().__init__(trainable=False, **kwargs)

        # Greedy MU-MIMO resource allocation strategy
        self.resource_allocation_strategy = "MU_MIMO"

        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 1

        # The number of transmitted streams is <= to the number of BS antennas
        self.num_streams_per_tx = 2
        self.num_subcarriers_per_RB = 12
        self.num_data_ofdm_symbols = 12
        self.num_data_ofdm_syms_per_RBG = 6

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = [[1]]

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        csi_effective_subcarriers = (cfg.fft_size // (self.ns3cfg.num_rxue_sel*self.ns3cfg.num_ue_ant)) * self.ns3cfg.num_rxue_sel*self.ns3cfg.num_ue_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        effective_subcarriers = cfg.fft_size - (cfg.fft_size % self.num_subcarriers_per_RB)
        while effective_subcarriers % (self.ns3cfg.num_rxue_sel * self.num_streams_per_tx) != 0:
            effective_subcarriers -= self.num_subcarriers_per_RB

        if effective_subcarriers <= 0:
            raise ValueError("Could not find a valid number of effective subcarriers with current config.")

        guard_carriers_1 = (cfg.fft_size - effective_subcarriers) // 2
        guard_carriers_2 = (cfg.fft_size - effective_subcarriers) - guard_carriers_1

        self.num_RBs = effective_subcarriers // self.num_subcarriers_per_RB
        self.num_RBGs_total = self.num_RBs * (self.num_data_ofdm_symbols / self.num_data_ofdm_syms_per_RBG)

        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=self.ns3cfg.num_rxue_sel*self.ns3cfg.num_ue_ant,
                                   cyclic_prefix_length=64,
                                   num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                                   dc_null=False,
                                   pilot_pattern="kronecker",
                                   pilot_ofdm_symbol_indices=[2, 11])

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=self.ns3cfg.num_rxue_sel,
                          num_streams_per_tx=self.num_streams_per_tx,
                          cyclic_prefix_length=64,
                          num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
        self.rg = rg

        self.h_freq_csi, err_var_csi = lmmse_channel_estimation(rxs_chans, self.rg_csi,
                                                           slot_idx=self.cfg.first_slot_idx-self.cfg.csi_delay,
                                                           cfo_vals=[0.0],
                                                           sto_vals=[0.0]) 
        self.perfect_channel_for_debugging, self.rx_snr_db, _ = rxs_chans.load_channel(slot_idx=self.cfg.first_slot_idx,
                                                                     batch_size=self.cfg.num_slots_p1)
        
        self.coderate = coderate  # fixed code rate for current design
        if self.resource_allocation_strategy == "MU_MIMO":

            self.num_UEs_per_RBG = 2 # This needs to be fixed for the current code
            assert self.num_UEs_per_RBG * self.num_streams_per_tx <= self.ns3cfg.num_bs_ant

            self.num_RBGs_per_UE = self.num_RBGs_total * self.num_UEs_per_RBG / self.ns3cfg.num_rxue_sel

            num_bits_per_stream_per_UE = np.ceil((rxs_bits_per_frame/self.ns3cfg.num_rxue_sel) 
                                                * (self.cfg.num_slots_p2 / self.cfg.num_slots_p1) 
                                                * (self.num_RBGs_total / self.num_RBGs_per_UE) 
                                                / self.num_streams_per_tx)
            
            bits_per_symbol_per_UE = int(np.ceil(num_bits_per_stream_per_UE / self.coderate / rg.num_data_symbols.numpy()))
            self.num_bits_per_symbol_per_UE = max(2, bits_per_symbol_per_UE)
            if self.num_bits_per_symbol_per_UE % 2 != 0:
                self.num_bits_per_symbol_per_UE += 1  # must be even
            assert self.num_bits_per_symbol_per_UE <= 12, "data bandwidth requires unsupported modulation order"

            self.MU_MIMO_RG_populated, self.precoding_matrices = self.mu_mimo_greedy_resource_allocator(rg)

        else:
            raise ValueError("Unsupported resource allocation strategy")

        # --- Grid and modulation parameters ---
        m = int(self.num_bits_per_symbol_per_UE)      # bits per QAM symbol, e.g., 12
        Nsym = int(rg.num_data_symbols.numpy())       # e.g., 6048
        num_streams = int(self.num_streams_per_tx)

        # --- How many bits the grid can carry ---
        grid_coded_capacity = self.num_UEs_per_RBG * num_streams * m * Nsym           # total coded bits mapped
        grid_info_capacity  = int(grid_coded_capacity * self.coderate)

        # --- Total bits using whole LDPC blocks (no segmentation math) ---
        self.total_coded_bits = int(grid_coded_capacity)
        self.total_info_bits  = int(grid_info_capacity)

        # --- Simple sanity checks ---
        assert self.total_coded_bits > self.total_info_bits
        assert rxs_bits_per_frame * self.cfg.num_slots_p2 <= self.total_info_bits

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", self.num_bits_per_symbol_per_UE)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(rg)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(rg, interpolation_type=None)

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", self.num_bits_per_symbol_per_UE)

        # The detector produces soft-decision output for all coded bits
        self.pic_detector = MMSEPICDetector("bit", rg, sm, constellation_type="qam",
                                        num_bits_per_symbol=self.num_bits_per_symbol_per_UE,
                                        num_iter=2, hard_out=False)
        
        self.precoding_method = 'mu_mimo_uplink_no_channel_reconstruction'
        
        if self.precoding_method == "mu_mimo_uplink_no_channel_reconstruction":
            self.phase_3_mu_mimo_uplink_precoder = phase_3_mu_mimo_uplink_precoder(rg, sm, architecture='phase_3')
        else:
            ValueError("unsupported precoding method")
        
        self.decoding_method = 'SIC_LMMSE'
        if self.decoding_method == "SIC_LMMSE":
            self.phase_3_mu_mimo_uplink_decoder = phase_3_sic_lmmse_decoder(rg, sm, architecture='phase_3')
        else:
            ValueError("unsupported decoding method")

        self._awgn = AWGN(dtype=tf.complex64)

    def call(self, rxs_chans, info_bits, min_codewords=8):
        """
        Signal processing for RxSquad downlink transmission (P1)

        :param rxs_chans: RxSquad channels
        :param info_bits: information bits
        :return: decoded bits, LDPC BER, LDPC BLER
        """

        # payload reshaping
        b = tf.transpose(info_bits, perm=[2, 0, 1, 3, 4])
        b = tf.reshape(b, [tf.shape(info_bits)[2], -1])

        m = int(self.num_bits_per_symbol_per_UE)
        U = self.ns3cfg.num_rxue_sel
        S = int(self.num_streams_per_tx)
        K = self.num_RBGs_per_UE * self.num_subcarriers_per_RB * self.num_data_ofdm_syms_per_RBG
        T = self.rg_csi.num_ofdm_symbols
        F   = int(self.rg_csi.fft_size)

        max_bits_per_ue = int(self.batch_size * m * K * S)

        assert b.shape[1] / self.coderate <= max_bits_per_ue

        self.num_codewords, self.ldpc_k, self.ldpc_n = self.plan_ldpc_blocks(max_bits_per_ue, b.shape[1], self.coderate, m, C_min=min_codewords, C_cap=64)

        assert self.ldpc_n*self.num_codewords >= max_bits_per_ue

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n, num_bits_per_symbol=m)
        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

        # Pad info bits to self.ldpc_k*self.num_codewords and make code blocks [U, self.num_codewords, self.ldpc_k]
        pad_len = self.ldpc_k * self.num_codewords - b.shape[1]
        assert pad_len >= 0
        pad_vals = tf.random.uniform([tf.shape(b)[0], pad_len], maxval=2, dtype=tf.int32)
        # Concatenate along the last dimension
        b = tf.concat([b, tf.cast(pad_vals,b.dtype)], axis=1)
        b_blocks   = tf.reshape(b, [U, self.num_codewords, self.ldpc_k])

        # LDPC encoding and interleaving
        c_blocks   = self.encoder(b_blocks)
        d = c_blocks

        # Concatenate CBs
        d_reshaped   = tf.reshape(d, [U, -1])
        assert d_reshaped.shape[-1] == max_bits_per_ue

        # Reshape to streams
        d_streams  = tf.reshape(d_reshaped, [self.batch_size, U, S, max_bits_per_ue//S])            # [B, U, S, max_bits_per_ue/S]

        # QAM mapping on OFDM grid. x has shape [batch_size, num_UEs, num_tx_streams_per_UE, num_qam_symbols_per_UE]
        x = self.mapper(d_streams)
        assert x.shape[-1] == self.num_RBGs_per_UE * self.num_subcarriers_per_RB * self.num_data_ofdm_syms_per_RBG
 
        # Starting with a dummy x_rg to place pilots. Will place in precoded data in precoder
        # TODO: there is an issue where it is possible that a precoder row is of form [v -v] and it gets multiplied by a stream column which is of shape [q q]^T. 
        # This will cause cancelling at a particular antenna
        d_dummy = tf.zeros([self.batch_size, U, S, m * int(self.rg.num_data_symbols.numpy())], dtype=tf.int32)
        x_dummy = self.mapper(d_dummy)
        x_dummy = tf.zeros(x_dummy.shape, dtype=x_dummy.dtype)
        x_rg_dummy = self.rg_mapper(x_dummy)
        x_precoded = self.phase_3_mu_mimo_uplink_precoder([x, x_rg_dummy, self.precoding_matrices, 
                                                                self.rg, self.precoding_method, self.MU_MIMO_RG_populated,
                                                                self.num_subcarriers_per_RB]) # [B, U, S, num_ofdm_syms, num_subcarriers]

        y_rg = tf.zeros([self.batch_size, 1, self.ns3cfg.num_bs_ant, T, F], dtype=x.dtype)
        h_hat = tf.zeros([self.batch_size, self.num_streams_per_tx, self.ns3cfg.num_bs_ant, T, F], dtype=x.dtype)
        for rx_ue_idx in range(0, self.ns3cfg.num_rxue_sel):

            tx_ant_mask = np.arange(2 * rx_ue_idx, 2 * rx_ue_idx + self.num_streams_per_tx)

            curr_x_precoded = x_precoded[:, rx_ue_idx, ...]

            curr_y_rg, _ = rxs_chans([curr_x_precoded, self.cfg.first_slot_idx, tx_ant_mask, None])

            y_rg += curr_y_rg

        y_rg = self.add_noise(y_rg, self.rx_snr_db)

        # LS channel estimation with linear interpolation
        no = 1e-5  # tunable param
        h_hat, err_var = self.ls_estimator([y_rg, no])
        h_hat = self.rb_wise_interpolate(h_hat)

        noise_var = 10.0 ** (-tf.reduce_mean(self.rx_snr_db)/10.0)
        noise_var = tf.cast(noise_var, h_hat.dtype.real_dtype)
        noise_var = tf.reduce_mean(noise_var)

        if self.decoding_method == "SIC_LMMSE":
            x_hat, no_eff = self.phase_3_mu_mimo_uplink_decoder([y_rg, h_hat, self.MU_MIMO_RG_populated, noise_var])
        else:
            raise ValueError("Unsupported decoder type {}".format(self.cfg.decoder))

        # Soft-output QAM demapper
        llr = self.demapper([x_hat, no_eff])

        # Hard-decision bit error rate
        d_hard = tf.cast(llr > 0, tf.float32)

        # Calculating the uncoded BER
        uncoded_ber = compute_ber(d_streams, d_hard).numpy()

        # Calculating the per-UE uncoded BER
        node_wise_uncoded_ber_list = []
        d_streams_reshaped = tf.reshape(d_streams, [self.batch_size, U, -1])
        d_hard_reshaped = tf.reshape(d_hard, [self.batch_size, U, -1])
        for ue_idx in range(U):
            node_wise_uncoded_ber_list.append(compute_ber(d_streams_reshaped[:, ue_idx, :], d_hard_reshaped[:, ue_idx, :]))
        node_wise_uncoded_ber  = tf.stack(node_wise_uncoded_ber_list)   # shape: [U]

        # LLR deinterleaver for LDPC decoding
        llr = tf.reshape(llr, [self.batch_size, U, -1])
        llr = tf.reshape(llr, [self.batch_size, U, self.num_codewords, self.ldpc_n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)
        dec_bits = tf.reshape(dec_bits, [self.batch_size, U, self.num_codewords*self.ldpc_k])

        # Remove the padding added before LDPC encoding to recover original info length per UE
        dec_bits = dec_bits[:, :, :b.shape[1]]
        dec_bits = tf.transpose(dec_bits, perm=[1, 0, 2])
        dec_bits = tf.reshape(dec_bits, [U, -1])

        # Reshaping into codeblocks
        b_blocks = tf.reshape(b, [U, self.num_codewords, -1])
        dec_bits_blocks = tf.reshape(dec_bits, [U, self.num_codewords, -1])

        # Coded BER and BLER Calculation
        coded_ber = compute_ber(b, dec_bits).numpy()
        coded_bler = compute_bler(b_blocks, dec_bits_blocks).numpy()

        # Per-node Coded BER and BLER Calculation
        node_wise_coded_ber_list = []
        node_wise_coded_bler_list = []
        for ue_idx in range(U):
            node_wise_coded_ber_list.append(compute_ber(b[ue_idx], dec_bits[ue_idx]))
            node_wise_coded_bler_list.append(compute_bler(b_blocks[ue_idx], dec_bits_blocks[ue_idx]))
        node_wise_coded_ber  = tf.stack(node_wise_coded_ber_list)   # shape: [U]
        node_wise_coded_bler = tf.stack(node_wise_coded_bler_list)  # shape: [U]


        return dec_bits, node_wise_uncoded_ber, uncoded_ber, node_wise_coded_ber, coded_ber, node_wise_coded_bler, coded_bler
    

    def rb_wise_interpolate(self, h_hat):

        per_rb_lin_interpolated_h_hat = tf.zeros(h_hat.shape, dtype=h_hat.dtype)
        for rb_idx in range(self.num_RBs):

            for pilot_sym_idx in range(self.rg.num_pilot_symbols // self.rg.num_effective_subcarriers):

                sc_idx = np.arange(rb_idx*self.num_subcarriers_per_RB, 
                    (rb_idx+1)*self.num_subcarriers_per_RB) + pilot_sym_idx*self.rg.num_effective_subcarriers
                
                curr_h_hat = tf.gather(h_hat, sc_idx, axis=-1)

                curr_h_hat_interp = self.linear_interpolate(curr_h_hat, dim=5)

                per_flat  = tf.reshape(per_rb_lin_interpolated_h_hat,  [-1, per_rb_lin_interpolated_h_hat.shape[-1]])   # [P, 1008]
                curr_flat = tf.reshape(curr_h_hat_interp, [-1, curr_h_hat_interp.shape[-1]])  # [P, 12]

                P = tf.shape(per_flat)[0]
                num_sc = tf.shape(curr_flat)[1]

                cols = tf.tile(tf.expand_dims(sc_idx, 0), [P,1])                    # [P,12]
                rows = tf.cast(tf.range(P)[:, None], cols.dtype)                    # [P,1]
                rows = tf.tile(rows, [1, num_sc])                                   # [P,12]
                idx2d = tf.stack([rows, cols], axis=-1)                             # [P,12,2]
                idx2d = tf.reshape(idx2d, [-1, 2])                                  # [P*12, 2]

                updates = tf.reshape(curr_flat, [-1])

                per_flat_updated = tf.tensor_scatter_nd_update(per_flat, idx2d, updates)

                per_rb_lin_interpolated_h_hat = tf.reshape(per_flat_updated, tf.shape(per_rb_lin_interpolated_h_hat))

        *prefix, last = per_rb_lin_interpolated_h_hat.shape.as_list()
        new_shape = prefix + [(self.rg.num_pilot_symbols // self.rg.num_effective_subcarriers).numpy(), self.rg.num_effective_subcarriers]
        per_rb_lin_interpolated_h_hat_reshaped = tf.reshape(per_rb_lin_interpolated_h_hat, new_shape)
        
        shape = per_rb_lin_interpolated_h_hat_reshaped.shape.as_list()
        shape[-2] = self.rg.num_ofdm_symbols
        ofdm_sym_lin_interpolated_h_hat = tf.zeros(shape, dtype=per_rb_lin_interpolated_h_hat_reshaped.dtype)

        idx = tf.stack(tf.meshgrid(*[tf.range(s) for s in ofdm_sym_lin_interpolated_h_hat.shape[:-2]], 
                                self.rg._pilot_ofdm_symbol_indices, 
                                tf.range(ofdm_sym_lin_interpolated_h_hat.shape[-1]), indexing='ij'), axis=-1)
        idx = tf.reshape(idx, [-1, idx.shape[-1]])
        ofdm_sym_lin_interpolated_h_hat = tf.tensor_scatter_nd_update(
            ofdm_sym_lin_interpolated_h_hat,
            idx,
            tf.reshape(per_rb_lin_interpolated_h_hat_reshaped, [-1])
        )

        interp_h_hat = self.linear_interpolate(
                   ofdm_sym_lin_interpolated_h_hat,
                   dim=5)

        return interp_h_hat

    def interpolate_batched(self, v):
        v = tf.convert_to_tensor(v)
        if v.dtype.is_complex:
            real_dtype = v.dtype.real_dtype
        else:
            real_dtype = v.dtype
        shape = tf.shape(v)
        B = shape[0]
        D = shape[1]
        x = tf.range(0, D, dtype=real_dtype)
        inf = tf.constant(float('inf'), dtype=real_dtype)
        zero = tf.zeros([], dtype=v.dtype)

        mask = tf.not_equal(v, zero)
        indices = tf.where(mask)  # [N, 2]
        if tf.size(indices) == 0:
            return tf.zeros_like(v)

        batch_indices = indices[:, 0]
        pos = tf.cast(indices[:, 1], real_dtype)
        fp = tf.gather_nd(v, indices)  # [N]

        xp_ragged = tf.RaggedTensor.from_value_rowids(
            pos, batch_indices, nrows=tf.cast(B, tf.int64))
        fp_ragged = tf.RaggedTensor.from_value_rowids(
            fp, batch_indices, nrows=tf.cast(B, tf.int64))

        num_points = tf.cast(xp_ragged.row_lengths(), tf.int32)  # [B]
        max_num = tf.reduce_max(num_points)
        min_num = tf.reduce_min(num_points)

        # assert min_num > 0

        xp_padded = xp_ragged.to_tensor(default_value=inf)   # [B, max_num]
        fp_padded = fp_ragged.to_tensor(default_value=zero)  # [B, max_num]

        result = tf.zeros_like(v)
        result = tf.tensor_scatter_nd_update(result, indices, fp)

        # Handle num_points == 1
        one_mask = tf.equal(num_points, 1)
        single_value = fp_padded[:, 0]
        result = tf.where(
            one_mask[:, tf.newaxis],
            single_value[:, tf.newaxis],
            result
        )

        # if *no* batch has >=2 points, we are done.
        # (max_num is the maximum row length across all batches.)
        if tf.less_equal(max_num, 1):
            return result

        # (you can now safely assume xp_padded and fp_padded have at least 2 columns)

        # Interpolation for >= 2
        left_idx = tf.searchsorted(
            xp_padded,
            tf.broadcast_to(x[tf.newaxis, :], [B, D]),
            side='right') - 1  # [B, D]

        left_idx_clip = tf.clip_by_value(
            left_idx,
            0,
            tf.maximum(0, num_points[:, tf.newaxis] - 2)
        )

        x_left = tf.gather(xp_padded, left_idx_clip, axis=1, batch_dims=1)
        f_left = tf.gather(fp_padded, left_idx_clip, axis=1, batch_dims=1)

        left_idx_clip_plus1 = left_idx_clip + 1
        x_right = tf.gather(xp_padded, left_idx_clip_plus1, axis=1, batch_dims=1)
        f_right = tf.gather(fp_padded, left_idx_clip_plus1, axis=1, batch_dims=1)

        frac = tf.cast((x[tf.newaxis, :] - x_left) / (x_right - x_left),
                    f_left.dtype)
        interp_val = f_left + frac * (f_right - f_left)

        in_between = tf.logical_and(
            left_idx >= 0,
            left_idx < num_points[:, tf.newaxis] - 1
        )
        gaps = tf.logical_and(in_between, x[tf.newaxis, :] > x_left)
        result = tf.where(gaps, interp_val, result)

        # Left extrapolation
        raw_slope_left = (fp_padded[:, 1] - fp_padded[:, 0]) / tf.cast(
            (xp_padded[:, 1] - xp_padded[:, 0]), v.dtype)
        slope_left = tf.where(num_points >= 2, raw_slope_left, zero)
        xp0 = xp_padded[:, 0]
        fp0 = fp_padded[:, 0]
        left_mask = x[tf.newaxis, :] < xp0[:, tf.newaxis]
        left_val = fp0[:, tf.newaxis] + tf.cast(
            (x[tf.newaxis, :] - xp0[:, tf.newaxis]), v.dtype) * slope_left[:, tf.newaxis]
        extrap_left_mask = tf.logical_and(left_mask, num_points[:, tf.newaxis] >= 2)
        result = tf.where(extrap_left_mask, left_val, result)

        # Right extrapolation
        last_idx = tf.maximum(0, num_points - 1)
        second_last_idx = tf.maximum(0, num_points - 2)
        xp_last = tf.gather(xp_padded, last_idx[:, tf.newaxis],
                            axis=1, batch_dims=1)[:, 0]
        fp_last = tf.gather(fp_padded, last_idx[:, tf.newaxis],
                            axis=1, batch_dims=1)[:, 0]
        xp_second_last = tf.gather(xp_padded, second_last_idx[:, tf.newaxis],
                                axis=1, batch_dims=1)[:, 0]
        fp_second_last = tf.gather(fp_padded, second_last_idx[:, tf.newaxis],
                                axis=1, batch_dims=1)[:, 0]
        raw_slope_right = (fp_last - fp_second_last) / tf.cast(
            (xp_last - xp_second_last), v.dtype)
        slope_right = tf.where(num_points >= 2, raw_slope_right, zero)
        right_mask = x[tf.newaxis, :] > xp_last[:, tf.newaxis]
        right_val = fp_last[:, tf.newaxis] + tf.cast(
            (x[tf.newaxis, :] - xp_last[:, tf.newaxis]), v.dtype) * slope_right[:, tf.newaxis]
        extrap_right_mask = tf.logical_and(
            right_mask, num_points[:, tf.newaxis] >= 2)
        result = tf.where(extrap_right_mask, right_val, result)

        return result


    def linear_interpolate(self, tensor, dim):
        tensor = tf.convert_to_tensor(tensor)
        ndim = tf.rank(tensor)
        # Permutation to move dim to the last axis
        perm = tf.concat([tf.range(0, dim), tf.range(dim + 1, ndim), [dim]], axis=0)
        transposed = tf.transpose(tensor, perm)
        shape = tf.shape(transposed)
        batch_shape = shape[:-1]
        D = shape[-1]
        num_batch = tf.math.reduce_prod(batch_shape)
        flat = tf.reshape(transposed, [num_batch, D])
        # Apply batched interpolation
        interp_flat = self.interpolate_batched(flat)
        # Reshape back
        interp_transposed = tf.reshape(interp_flat, tf.concat([batch_shape, [D]], axis=0))
        # Inverse permutation
        inv_perm = tf.argsort(perm)
        result = tf.transpose(interp_transposed, inv_perm)
        return result
    
    
    def plan_ldpc_blocks(self, E_total, info_total, coderate, m, C_min=4, C_cap=64):
        """
        Prefer fewer (larger) codeblocks.
        Constraints:
        C * n_cb = E_total
        n_cb % m == 0
        ceil(info_total / C) / n_cb <= coderate
        Returns (C, k_cb, n_cb).
        """

        # Upper bound on how many CBs you can even have given the m-multiple constraint
        max_C = min(int(C_cap), int(E_total // m))
        best = None

        # Prefer fewer CBs: scan upward from C=1
        for C in range(C_min, max_C + 1):
            if E_total % C != 0:
                continue
            n_cb = E_total // C
            if n_cb % m != 0:
                continue

            # Info per CB needed to carry info_total across C blocks
            k_cb = int(np.ceil(info_total / C))

            # Check nominal rate feasibility (with tiny tolerance)
            if k_cb <= int(np.floor(coderate * n_cb + 1e-9)):
                best = (C, k_cb, n_cb)
                break  # first match uses the fewest CBs

        if best is None:
            # Fallback: keep n_cb a multiple of m near E_total, accept slack handling upstream if needed
            C = 1
            n_cb = (E_total // m) * m
            k_cb = int(np.ceil(info_total / C))
            best = (C, k_cb, n_cb)

        return best




    def choose_ldpc_n(self, info_bits, max_coded_bits):
        
        VALID_N = [8448, 8064, 7680, 7392, 7104, 6912, 6720, 6336, 6144, 5952, 5760]

        for n in VALID_N:
            k = int(n * self.coderate)
            
            # Blocks needed to fill (and slightly exceed) the grid, so we can puncture back
            blocks = int(np.ceil(max_coded_bits / n))
            
            return k, n, blocks
        raise ValueError("No (k,n) fits grid capacity")
    
    def add_noise(self, y, rx_snr_db):
        
        no = tf.cast(np.power(10.0, rx_snr_db / (-10.0)), tf.float32)
        no = tf.expand_dims(no, -1)  # [batch_size, num_rx, num_rx_ant, num_ofdm_sym, 1]
        y = self._awgn([y, no])

        return y


    def orthonormalize_cols(self, A):
        # QR gives orthonormal columns in Q (up to rank)
        Q, _ = np.linalg.qr(A)
        return Q[:, :A.shape[1]]

    def chordal_distance(self, U, V):
        """
        U, V: (m, r) with orthonormal columns.
        Works for real or complex.
        d^2 = r - ||U^* V||_F^2
        """
        r = U.shape[1]
        gram = U.conj().T @ V
        return np.sqrt(max(r - np.linalg.norm(gram, 'fro')**2, 0.0))

    def farthest_pair_by_chordal(self, arr, remaining_UEs_to_allocate):
        """
        arr: shape (N, m, r). Columns are (m x r) subspace bases per UE.
        remaining_UEs_to_allocate: np.array of indices in the *global* UE space,
            but `arr` is already sliced to this subset. We only use the length:
            - if len == 2: return (0, 1) immediately
            - if len > 2: compute over arr normally

        Returns:
            (i, j), D
            where (i, j) are indices into axis-0 of `arr` (local subset),
            and D is the symmetric NxN matrix of chordal distances among rows of `arr`.
        """
        # Handle small-N cases upfront
        N = arr.shape[0]
        if N < 2:
            raise ValueError("Need at least two candidates in `arr`.")

        # Fast path when exactly two remain: pair is trivially (0, 1)
        if len(remaining_UEs_to_allocate) == 2:
            # Still build D for consistency
            U0 = self.orthonormalize_cols(arr[0])
            U1 = self.orthonormalize_cols(arr[1])
            gram = U0.conj().T @ U1
            r = U0.shape[1]
            d = float(np.sqrt(max(r - np.linalg.norm(gram, ord='fro')**2, 0.0)))
            D = np.zeros((2, 2), dtype=float)
            D[0, 1] = D[1, 0] = d
            return (0, 1), D

        # General path: compute pair with max chordal distance within this subset
        _, m, r = arr.shape
        U = [self.orthonormalize_cols(arr[k]) for k in range(N)]
        D = np.zeros((N, N), dtype=float)

        best_d = -1.0
        best_pair = (0, 1)  # safe default

        for i in range(N):
            Ui = U[i]
            for j in range(i + 1, N):
                Uj = U[j]
                gram = Ui.conj().T @ Uj
                d2 = r - np.linalg.norm(gram, ord='fro')**2
                d = float(np.sqrt(max(d2, 0.0)))
                D[i, j] = D[j, i] = d
                if d > best_d:
                    best_d = d
                    best_pair = (i, j)

        return best_pair, D
    
    def mu_mimo_greedy_resource_allocator(self, rg):


        """
            Model:
                                                     time → (14 OFDM symbols)
                         ┌───────────────────────────────────────────────────────────────────────────────┐
                 RB 0    │                              [ UE 0 ]   [ UE 1 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                 RB 1    │                              [ UE 1 ]   [ UE 2 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                 RB 2    │                              [ UE 0 ]   [ UE 2 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                 RB 3    │                              [ UE 0 ]   [ UE 1 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                   .     │                              [ UE 1 ]   [ UE 2 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
              freq  ↓    │                              [ UE 1 ]   [ UE 2 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                   .     │                              [ UE 0 ]   [ UE 2 ]                              │
                         ├───────────────────────────────────────────────────────────────────────────────┤
                   .     │                              [ UE 0 ]   [ UE 1 ]                              │
                         └───────────────────────────────────────────────────────────────────────────────┘
                                                            --------→       
                                                            OFDM symbols

        """

        generate_CSI_feedback = quantized_CSI_feedback(method='5G', codebook_selection_method='rate',
                                                       num_tx_streams=self.num_streams_per_tx, 
                                                       architecture='dMIMO_phase3_SU_MIMO', snrdb=self.rx_snr_db,
                                                       wideband=True)
        
        precoding_matrices_all = np.zeros([self.ns3cfg.num_rxue_sel, 14, rg.fft_size, self.ns3cfg.num_ue_ant, self.num_streams_per_tx], dtype=complex)
        
        MU_MIMO_RG_populated = -1 * np.ones([self.num_RBs, (self.num_data_ofdm_symbols // self.num_data_ofdm_syms_per_RBG), self.num_UEs_per_RBG], np.int64)
        remaining_UEs_to_allocate = np.arange(self.ns3cfg.num_rxue_sel)
        remaining_UEs_to_allocate_in_each_step = []
        D_in_each_step = []
        remaining_RBGs_per_UE = np.ones(self.ns3cfg.num_rxue_sel) * self.num_RBGs_per_UE // 2
        for rb_idx in range(self.num_RBs):
            sc_idx = np.arange(rg.num_guard_carriers[0] + rb_idx*self.num_subcarriers_per_RB, 
                            rg.num_guard_carriers[0] + (rb_idx+1)*self.num_subcarriers_per_RB)
            curr_h_freq_csi = tf.gather(self.h_freq_csi, sc_idx, axis=-1)
            [_, _, precoding_matrices] = generate_CSI_feedback(curr_h_freq_csi)
            precoding_matrices_all[:, :, sc_idx, :, :] = precoding_matrices
            
            # RB Mapping (Assuming self.num_subcarriers_per_RB subcarriers per RB)
            # TPMI is calculated only once per RBG. 
            curr_h_freq_csi_RB = np.mean(curr_h_freq_csi,axis=-1)
            curr_h_freq_csi_RB = np.mean(curr_h_freq_csi_RB, axis=-1)
            curr_h_freq_csi_RB = np.transpose(curr_h_freq_csi_RB, [0,1,3,2,4])

            curr_h_freq_csi_RB_effective = np.zeros((len(remaining_UEs_to_allocate), self.ns3cfg.num_bs_ant, self.ns3cfg.num_ue_ant), dtype=complex)
            for ue_idx_relative, ue_idx_absolute in enumerate(remaining_UEs_to_allocate):
                ue_ant_idx = np.arange(self.ns3cfg.num_ue_ant*ue_idx_absolute, self.ns3cfg.num_ue_ant*(ue_idx_absolute+1))
                curr_h_freq_csi_RB_effective[ue_idx_relative, ...] = np.squeeze(curr_h_freq_csi_RB[...,ue_ant_idx]) @ np.squeeze(precoding_matrices[ue_idx_relative, ...])

            remaining_UEs_to_allocate_in_each_step.append(remaining_UEs_to_allocate)

            for ue_idx in range(self.ns3cfg.num_rxue_sel): 
                if np.sum(MU_MIMO_RG_populated == ue_idx) == self.num_RBGs_per_UE:
                    hold = 1

            ############################ Resource allocation ######################################
            if len(remaining_UEs_to_allocate) > 1:
                UE_pair_scheduled, D = self.farthest_pair_by_chordal(curr_h_freq_csi_RB_effective, remaining_UEs_to_allocate)
                D_in_each_step.append(D)
                UE_pair_scheduled = remaining_UEs_to_allocate[[UE_pair_scheduled]][0]
                MU_MIMO_RG_populated[rb_idx, 0, :] = np.asarray(UE_pair_scheduled)
                MU_MIMO_RG_populated[rb_idx, 1, :] = np.asarray(UE_pair_scheduled)

                remaining_RBGs_per_UE[UE_pair_scheduled[0]] -= 1
                if remaining_RBGs_per_UE[UE_pair_scheduled[0]] == 0:
                    remaining_UEs_to_allocate = remaining_UEs_to_allocate[remaining_UEs_to_allocate != UE_pair_scheduled[0]]
                
                remaining_RBGs_per_UE[UE_pair_scheduled[1]] -= 1
                if remaining_RBGs_per_UE[UE_pair_scheduled[1]] == 0:
                    remaining_UEs_to_allocate = remaining_UEs_to_allocate[remaining_UEs_to_allocate != UE_pair_scheduled[1]]
            else:
                MU_MIMO_RG_populated[rb_idx, 0, :] = np.asarray([remaining_UEs_to_allocate[0], -1])
                MU_MIMO_RG_populated[rb_idx, 1, :] = np.asarray([remaining_UEs_to_allocate[0], -1])
                D_in_each_step.append(0)


        # ---- Second pass: fill holes and swap to satisfy victim quota ----
        U      = self.ns3cfg.num_rxue_sel
        quota  = int(self.num_RBGs_per_UE)

        # helper to get D and "remaining set" for a (rb, side)
        def step_D_and_set(rb):
            D = D_in_each_step[:rb]
            rem = remaining_UEs_to_allocate_in_each_step[:rb]
            return D, rem

        # current counts per UE
        flat = MU_MIMO_RG_populated[MU_MIMO_RG_populated >= 0].astype(int)
        counts = np.bincount(flat, minlength=U)
        victims = np.where(counts < quota)[0]

        for victim in victims:
            # iterate holes; fill with best donor for that step
            for rb in range(self.num_RBs):
                if MU_MIMO_RG_populated[rb, 0, 1] != -1:
                    continue  # not a hole
                D, rem = step_D_and_set(rb)

                i = 0
                while(True):
                    if victim not in MU_MIMO_RG_populated[i,0,:]:
                        donor_candidates = (np.unique(MU_MIMO_RG_populated[i,...]))
                        donor_candidate_chordal_distances = D[i][victim, :]
                        donor_candidate_chordal_distances = donor_candidate_chordal_distances[donor_candidates]

                        donor = donor_candidates[donor_candidate_chordal_distances == max(donor_candidate_chordal_distances)]

                        MU_MIMO_RG_populated[rb, 0, 1] = donor[0]
                        MU_MIMO_RG_populated[rb, 1, 1] = donor[0]
                        MU_MIMO_RG_populated[i, 0, np.where(MU_MIMO_RG_populated[i, 0, :] == donor)[0]] = victim
                        MU_MIMO_RG_populated[i, 1, np.where(MU_MIMO_RG_populated[i, 1, :] == donor)[0]] = victim
                        break
                    else:
                        i += 1   

        # Final check
        for ue_idx in range(self.ns3cfg.num_rxue_sel): 
            assert np.sum(MU_MIMO_RG_populated == ue_idx) == self.num_RBGs_per_UE

        return MU_MIMO_RG_populated, precoding_matrices_all


