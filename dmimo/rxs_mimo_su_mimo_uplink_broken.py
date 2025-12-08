import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.ofdm import LMMSEEqualizer, MMSEPICDetector
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import SimConfig, Ns3Config
from dmimo.channel import lmmse_channel_estimation
from dmimo.mimo import quantized_CSI_feedback, fiveGPrecoder

class RxSquad(Model):
    """
    Implement Rx Squad data transmission in phase 3 (P3)

    For the Rx squad uplink, OFDMA is required for more than 2 UEs. Here we simulate
    the average performance for each pair of UEs uplink across all subcarriers, i.e.,
    two UEs are transmitting simultaneously to the BS with 4x4 MIMO channels.

    """

    def __init__(self, cfg: SimConfig, ns3cfg: Ns3Config, rxs_bits_per_frame: int, **kwargs):
        """
        Initialize RxSquad simulation

        :param cfg: simulation settings
        :param rxs_bits_per_frame: total number of bits per subframe/slot for all UEs
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.ns3cfg = ns3cfg
        self.batch_size = cfg.num_slots_p1  # batch processing for all slots in phase 1

        # The number of transmitted streams is equal to the number of BS antennas
        num_ue_ant = 2
        self.num_streams_per_tx = 1

        # Doing SU-MIMO resource allocation equally between UEs to start with
        self.resource_allocation_strategy = "SU_MIMO_round_robin"
        self.num_subcarriers_per_RB = 12
        self.num_data_ofdm_symbols = 12
        self.num_data_ofdm_syms_per_RBG = 6
        if self.resource_allocation_strategy == "SU_MIMO_round_robin":
            num_RBs_per_sym_per_UE = np.floor(cfg.fft_size / (self.num_subcarriers_per_RB * self.ns3cfg.num_rxue_sel))
            num_RBs_per_sym_total = num_RBs_per_sym_per_UE * self.ns3cfg.num_rxue_sel
            num_used_subcarriers = num_RBs_per_sym_per_UE * self.num_subcarriers_per_RB * self.ns3cfg.num_rxue_sel
            assert num_used_subcarriers <= cfg.fft_size
            assert np.mod(self.num_data_ofdm_symbols, self.num_data_ofdm_syms_per_RBG) == 0 # this is an assumption
            self.num_RBGs_per_UE = num_RBs_per_sym_per_UE * self.num_data_ofdm_symbols // self.num_data_ofdm_syms_per_RBG
            self.num_RBGs_total = int(num_RBs_per_sym_total * self.num_data_ofdm_symbols // self.num_data_ofdm_syms_per_RBG)
        else:
            raise 

        # AdHoc way of making the num_RBGs_per_UE a whole number
        residual_subcarriers = int(cfg.fft_size - (num_RBs_per_sym_total * self.num_subcarriers_per_RB))

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = [[1]]

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        csi_effective_subcarriers = (cfg.fft_size // (self.ns3cfg.num_rxue_sel*num_ue_ant)) * self.ns3cfg.num_rxue_sel*num_ue_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

        effective_subcarriers = (csi_effective_subcarriers // self.num_streams_per_tx) * self.num_streams_per_tx
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += (residual_subcarriers // 2)
        guard_carriers_2 += (residual_subcarriers - residual_subcarriers//2)

        self.rg_csi = ResourceGrid(num_ofdm_symbols=14,
                                   fft_size=cfg.fft_size,
                                   subcarrier_spacing=cfg.subcarrier_spacing,
                                   num_tx=1,
                                   num_streams_per_tx=self.ns3cfg.num_rxue_sel*num_ue_ant,
                                   cyclic_prefix_length=64,
                                   num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                                   dc_null=False,
                                   pilot_pattern="kronecker",
                                   pilot_ofdm_symbol_indices=[2, 11])

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=self.num_streams_per_tx,
                          cyclic_prefix_length=64,
                          num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])        

        # Doing round-robin UE allocation to start with
        assert self.num_RBGs_total % self.ns3cfg.num_rxue_sel == 0, "Round-robin needs number of total RBGs divisible by number of UEs"
        assert int(rg.num_data_symbols.numpy()) % self.num_RBGs_total == 0, "Need Nsym divisible by num_RBGs_total"

        syms_per_RBG = int(rg.num_data_symbols.numpy()) // self.num_RBGs_total
        self.syms_per_UE = int(rg.num_data_symbols.numpy()) // self.ns3cfg.num_rxue_sel

        # For each RBG index g, owner UE is g % U; its symbol range is g*syms_per_RBG : (g+1)*syms_per_RBG
        idxs = [ [] for _ in range(self.ns3cfg.num_rxue_sel) ]
        for g in range(self.num_RBGs_total):
            u = g % self.ns3cfg.num_rxue_sel
            start = g * syms_per_RBG
            stop  = start + syms_per_RBG
            idxs[u].extend(range(start, stop))

        idx_per_ue = np.stack([np.array(idxs[u], dtype=np.int32) for u in range(self.ns3cfg.num_rxue_sel)], axis=0)  # [U, K]
        self.round_robin_indices = tf.constant(idx_per_ue, dtype=tf.int32)

        self.coderate = 5/6  # fixed code rate for current design
        num_bits_per_stream_per_UE = np.ceil((rxs_bits_per_frame/self.ns3cfg.num_rxue_sel) * (cfg.num_slots_p2 / cfg.num_slots_p1) * (self.num_RBGs_total / self.num_RBGs_per_UE) / self.num_streams_per_tx)
        bits_per_symbol_per_UE = int(np.ceil(num_bits_per_stream_per_UE / self.coderate / rg.num_data_symbols.numpy()))
        self.num_bits_per_symbol_per_UE = max(2, bits_per_symbol_per_UE)
        if self.num_bits_per_symbol_per_UE % 2 != 0:
            self.num_bits_per_symbol_per_UE += 1  # must be even
        self.num_bits_per_frame_per_UE = int(rg.num_data_symbols.numpy() * self.num_bits_per_symbol_per_UE * self.coderate * self.num_streams_per_tx)
        assert self.num_bits_per_symbol_per_UE <= 12, "data bandwidth requires unsupported modulation order"

        # --- Grid and modulation parameters ---
        m = int(self.num_bits_per_symbol_per_UE)      # bits per QAM symbol, e.g., 12
        Nsym = int(rg.num_data_symbols.numpy())       # e.g., 6048
        num_streams = int(self.num_streams_per_tx)

        # --- How many bits the grid can carry ---
        grid_coded_capacity = m * Nsym * num_streams          # total coded bits mapped
        grid_info_capacity  = int(grid_coded_capacity * self.coderate)

        # --- Total bits using whole LDPC blocks (no segmentation math) ---
        self.total_coded_bits_per_UE = int(grid_coded_capacity)
        self.total_info_bits_per_UE  = int(grid_info_capacity)

        # --- Simple sanity check ---
        assert self.total_coded_bits_per_UE > self.total_info_bits_per_UE

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", self.num_bits_per_symbol_per_UE)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(rg)

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(rg, interpolation_type="lin")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", self.num_bits_per_symbol_per_UE)

        # The detector produces soft-decision output for all coded bits
        self.pic_detector = MMSEPICDetector("bit", rg, sm, constellation_type="qam",
                                        num_bits_per_symbol=self.num_bits_per_symbol_per_UE,
                                        num_iter=2, hard_out=False)
        
        self.precoding_method = '5G_no_channel_reconstruction'
        if "5G" in self.precoding_method:
            # The 5G SU MIMO precoder
            self.fiveG_precoder = fiveGPrecoder(rg, sm, architecture='dMIMO_phase3_SU_MIMO')
        else:
            ValueError("unsupported precoding method")

        self.data_coords = self.find_data_coords(rg)


    def call(self, rxs_chans, info_bits, return_data_only=False):
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
        K = self.syms_per_UE
        T = self.rg_csi.num_ofdm_symbols
        F   = int(self.rg_csi.fft_size)

        max_bits_per_ue = m * K * S

        assert b.shape[1] / self.coderate <= max_bits_per_ue

        per_stream_budget = m * K
        self.ldpc_k, self.ldpc_n, self.num_codewords = self.choose_ldpc_n(b.shape[1], per_stream_budget)
        tf.debugging.assert_equal(self.ldpc_n * self.num_codewords, per_stream_budget)

        assert self.ldpc_n * self.num_codewords == max_bits_per_ue

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)
        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

        # Random 0/1 padding
        pad_len = self.ldpc_k * self.num_codewords - b.shape[1]
        assert pad_len >= 0
        pad_vals = tf.random.uniform([tf.shape(b)[0], pad_len], maxval=2, dtype=tf.int32)
        # Concatenate along the last dimension
        b = tf.concat([b, tf.cast(pad_vals,b.dtype)], axis=1)
        b_blocks   = tf.reshape(b, [U, self.num_codewords, self.ldpc_k])

        # LDPC encoding and interleaving
        c_blocks   = self.encoder(b_blocks)
        c     = tf.reshape(c_blocks, [self.batch_size, U, self.num_streams_per_tx, self.num_codewords * self.encoder.n])
        d = self.intlvr(c)

        # QAM mapping on OFDM grid. x has shape [batch_size, num_UEs, num_tx_streams_per_UE, num_qam_symbols_per_UE]
        x = self.mapper(d)

        x_interleaved = tf.zeros([self.batch_size, 1, S, U * K], dtype=x.dtype)
        for u in range(U):
            ii_batch = tf.repeat(tf.range(self.batch_size)[:, None], K, axis=1)
            ii_tx    = tf.zeros_like(ii_batch)                 # tx = 0
            pos_u    = tf.tile(self.round_robin_indices[u][None, :], [self.batch_size, 1])        # UE u's target positions

            for s in range(S):
                ii_strm = tf.fill([self.batch_size, K], s)
                inds = tf.stack([ii_batch, ii_tx, ii_strm, pos_u], axis=-1)  # [B,K,4]
                inds = tf.reshape(inds, [-1, 4])                              # [B*K,4]

                updates = tf.reshape(x[:, u, s, :], [-1])                     # [B*K]
                x_interleaved = tf.tensor_scatter_nd_update(x_interleaved, inds, updates)

        # x_rg has shape [batch_size, num_tx, num_tx_streams, num_ofdm_syms, fft_size]
        x_rg = self.rg_mapper(x_interleaved)

        if return_data_only:
            return x_rg

        h_freq_csi, err_var_csi = lmmse_channel_estimation(rxs_chans, self.rg_csi,
                                                           slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                           cfo_vals=[0.0],
                                                           sto_vals=[0.0])
        _, rx_snr_db, _ = rxs_chans.load_channel(slot_idx=self.cfg.first_slot_idx - self.cfg.csi_delay,
                                                                     batch_size=self.cfg.num_slots_p1)
        generate_CSI_feedback = quantized_CSI_feedback(method='5G', codebook_selection_method='rate',
                                                       num_tx_streams=self.num_streams_per_tx, 
                                                       architecture='dMIMO_phase3_SU_MIMO', snrdb=rx_snr_db)
        [TPMI, rate_for_selected_precoder, precoding_matrices] = generate_CSI_feedback(h_freq_csi)

        # check all UEs
        ue_data = []
        ue_ber_avg, ue_bler_avg = 0.0, 0.0
        ue_ber_max, ue_bler_max = 0.0, 0.0

        y_rg = tf.zeros([self.batch_size, 1, self.ns3cfg.num_bs_ant, T, F], dtype=x_rg.dtype)
        h_hat = tf.zeros([self.batch_size, self.num_streams_per_tx, self.ns3cfg.num_bs_ant, T, F], dtype=x_rg.dtype)
        for rx_ue_idx in range(0, self.ns3cfg.num_rxue_sel):

            if "5G" in self.precoding_method:
                curr_x_precoded = self.fiveG_precoder([x_rg, precoding_matrices[rx_ue_idx,...], self.precoding_method])
                pad_sizes = [[0,0],
                            [0,0],
                            [0,(self.ns3cfg.num_rxue_sel-1)*2],
                            [0,0],
                            [0,0]]
                curr_x_precoded = tf.pad(curr_x_precoded, pad_sizes)

            else:
                ValueError("unsupported precoding method")

            curr_y_rg, _ = rxs_chans([curr_x_precoded, self.cfg.first_slot_idx])

            # LS channel estimation with linear interpolation
            no = 1e-5  # tunable param
            curr_h_hat, err_var = self.ls_estimator([curr_y_rg, no])
            curr_h_hat = tf.squeeze(curr_h_hat, axis=[1,3])

            lin_idx = self.round_robin_indices[rx_ue_idx]
            coords_u = tf.gather(self.data_coords, lin_idx)

            mask2d_bool = tf.scatter_nd(
                indices=coords_u,                          # [K,2]
                updates=tf.ones([K], dtype=tf.bool),       # True at UE-owned REs
                shape=[T, F]
            )                                              # [T, F] bool

            # Broadcast to [B,1,Nb,T,F]
            mask5d_bool = tf.reshape(mask2d_bool, [1,1,1,T,F])    # bool
            mask5d_bool = tf.tile(mask5d_bool, [self.batch_size,1,self.ns3cfg.num_bs_ant,1,1])      # bool

            # Overwrite only at UE-owned REs (curr_y_rg, y_rg are complex64)
            y_rg = tf.where(mask5d_bool, curr_y_rg, y_rg)

            mask_chan = tf.tile(mask5d_bool[:, :, :1, :, :], [1, 1, self.ns3cfg.num_bs_ant, 1, 1])
            h_hat = tf.where(mask_chan, curr_h_hat, h_hat)


         



        if self.cfg.decoder == "pic":

            raise ValueError("Unsupported decoder type {}".format(self.cfg.decoder))
            prior = tf.zeros(d.shape)
            det_out = self.pic_detector((y_rg, h_hat, prior, err_var, no))

            # LLR interleaving
            llr = tf.reshape(det_out, d.shape)
            llr = self.dintlvr(llr)

            # LDPC decoder
            llr = tf.reshape(llr, [self.batch_size, 1, self.num_streams_per_tx, self.num_codewords, self.encoder.n])
            dec_bits = self.decoder(llr)
        elif self.cfg.decoder == "lmmse":
            x_hat, no_eff = self.lmmse_equ([y_rg, h_hat, err_var, no])
            llr = self.demapper([x_hat, no_eff])
            dec_bits = tf.cast(llr > 0, tf.float32)
        else:
            raise ValueError("Unsupported decoder type {}".format(self.cfg.decoder))

            # Error statistics
            ber = compute_ber(b, dec_bits).numpy()
            bler = compute_bler(b, dec_bits).numpy()
            ue_ber_avg += ber/self.ns3cfg.num_rxue_sel
            ue_bler_avg += bler/self.ns3cfg.num_rxue_sel
            if ber > ue_ber_max:
                ue_ber_max = ber
            if bler > ue_bler_max:
                ue_bler_max = bler
            ue_data = dec_bits

        # Remove padding bits
        if self.ldpc_padding > 0:
            ue_data = tf.reshape(ue_data, (-1))
            ue_data = ue_data[:-self.ldpc_padding]
        # restore original shape
        ue_data = tf.reshape(ue_data, (self.cfg.num_slots_p2, -1))

        return ue_data, ue_ber_avg, ue_bler_avg, ue_ber_max, ue_bler_max


    def choose_ldpc_n(self, info_bits, max_coded_bits):
        
        VALID_N = [8448, 8064, 7680, 7392, 7104, 6912, 6720, 6336, 6144, 5952, 5760]

        for n in VALID_N:
            k = int(n * self.coderate)
            blocks = int(np.ceil(info_bits / k))
            if blocks * n <= max_coded_bits:
                return k, n, blocks
        raise ValueError("No (k,n) fits grid capacity; increase m/K/S or reduce info_bits.")

    def find_data_coords(self, rg):
        T   = int(rg.num_ofdm_symbols)               # 14
        F   = int(rg.fft_size)
        Nsym = int(rg.num_data_symbols.numpy())      # per (tx,stream)
        S   = int(rg.num_streams_per_tx)             # 1 or 2

        # Build a probe with S streams; put a "ramp" only on stream 0
        ramp = tf.range(1, Nsym+1, dtype=tf.int32)[None, None, None, :]      # [1,1,1,Nsym]
        ramp = tf.cast(ramp, tf.complex64)                                   # complex
        if S == 1:
            probe = ramp                                                     # [1,1,1,Nsym]
        else:
            zeros_other = tf.zeros([1,1,S-1,Nsym], dtype=tf.complex64)       # [1,1,S-1,Nsym]
            probe = tf.concat([ramp, zeros_other], axis=2)                   # [1,1,S,Nsym]

        # Map into the grid: shape [1, 1, S, T, F]
        grid = self.rg_mapper(probe)

        # Take stream 0 (the only stream with non-zero ramp)
        grid0 = grid[:, 0, 0, :, :]                                          # [1, T, F]
        grid_i = tf.reshape(tf.cast(tf.math.real(grid0), tf.int32), [T*F])   # [T*F]

        # Positions where data was placed (values > 0)
        flat_pos = tf.where(grid_i > 0)[:, 0]                                # [Nsym]
        vals     = tf.gather(grid_i, flat_pos) - 1                           # linear 0..Nsym-1

        # Convert flat positions to (t, f)
        t_coords = flat_pos // F
        f_coords = flat_pos %  F
        coords_unsorted = tf.stack([t_coords, f_coords], axis=1)             # [Nsym, 2]

        # coords[lin] = (t,f) in the mapper's linear order
        coords_init = tf.zeros([Nsym, 2], dtype=tf.int32)
        data_coords = tf.tensor_scatter_nd_update(
            coords_init,
            tf.expand_dims(vals, axis=1),   # [Nsym,1]
            tf.cast(coords_unsorted, coords_init.dtype)
        )
        return data_coords
