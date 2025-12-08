import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
import time

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, flatten_dims
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig, RCConfig
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation
from dmimo.channel import standard_rc_pred_freq_mimo
from dmimo.channel import twomode_wesn_pred
from dmimo.channel import RBwiseLinearInterp
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, SLNREqualizer, QuantizedZFPrecoder, QuantizedDirectPrecoder
from dmimo.mimo import rankAdaptation, linkAdaptation
from dmimo.mimo import MUMIMOScheduler
from dmimo.mimo import update_node_selection, quantized_CSI_feedback, RandomVectorQuantizer, RandomVectorQuantizerNumpy
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad


class MU_MIMO(Model):

    def __init__(self, cfg: SimConfig, rg_csi: ResourceGrid, **kwargs):
        """
        Create MU-MIMO simulation object

        :param cfg: simulation settings
        :param rg_csi: Resource grid for CSI estimation
        """
        super().__init__(trainable=False, **kwargs)

        self.cfg = cfg
        self.rg_csi = rg_csi
        self.batch_size = cfg.num_slots_p2  # batch processing for all slots in phase 2

        # To use sionna-compatible interface, regard TxSquad as one BS transmitter
        # A 4-antennas basestation is regarded as the combination of two 2-antenna UEs
        self.num_streams_per_tx = cfg.num_tx_streams

        self.num_ue_ant = 2  # assuming 2 antennas per UE for reshaping data/channels
        if cfg.ue_indices is None:
            # no rank/link adaptation
            self.num_rxs_ant = self.num_streams_per_tx
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
        else:
            # rank adaptation support
            self.num_rxs_ant = np.sum([len(val) for val in cfg.scheduled_rx_ue_indices])
            self.num_rx_ue = self.num_rxs_ant // self.num_ue_ant
            if cfg.ue_ranks is None:
                cfg.ue_ranks = self.num_ue_ant  # no rank adaptation

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream from transmitter j.
        rx_tx_association = np.ones((self.num_rx_ue, 1))

        # Instantiate a StreamManagement object
        # This determines which data streams are determined for which receiver.
        sm = StreamManagement(rx_tx_association, self.num_streams_per_tx)

        # Adjust guard subcarriers for different number of streams
        csi_effective_subcarriers = self.rg_csi.num_effective_subcarriers
        csi_guard_carriers_1 = self.rg_csi.num_guard_carriers[0]
        csi_guard_carriers_2 = self.rg_csi.num_guard_carriers[1]
        effective_subcarriers = (csi_effective_subcarriers // self.num_streams_per_tx) * self.num_streams_per_tx
        guard_carriers_1 = (csi_effective_subcarriers - effective_subcarriers) // 2
        guard_carriers_2 = (csi_effective_subcarriers - effective_subcarriers) - guard_carriers_1
        guard_carriers_1 += csi_guard_carriers_1
        guard_carriers_2 += csi_guard_carriers_2

        # OFDM resource grid (RG) for normal transmission
        self.rg = ResourceGrid(num_ofdm_symbols=14,
                               fft_size=cfg.fft_size,
                               subcarrier_spacing=cfg.subcarrier_spacing,
                               num_tx=1,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=64,
                               num_guard_carriers=[guard_carriers_1, guard_carriers_2],
                               dc_null=False,
                               pilot_pattern="kronecker",
                               pilot_ofdm_symbol_indices=[2, 11])

        # Update number of data bits and LDPC params
        self.ldpc_n = int(2 * self.rg.num_data_symbols)  # Number of coded bits
        self.ldpc_k = int(self.ldpc_n * cfg.code_rate)  # Number of information bits
        self.num_codewords = cfg.modulation_order // 2  # number of codewords per frame
        self.num_bits_per_frame = self.ldpc_k * self.num_codewords * self.num_streams_per_tx
        self.num_uncoded_bits_per_frame = self.ldpc_n * self.num_codewords * self.num_streams_per_tx

        # The encoder maps information bits to coded bits
        self.encoder = LDPC5GEncoder(self.ldpc_k, self.ldpc_n)

        # LDPC interleaver
        self.intlvr = RowColumnInterleaver(3072, axis=-1)  # fixed design for current RG config
        self.dintlvr = Deinterleaver(interleaver=self.intlvr)

        # The mapper maps blocks of information bits to constellation symbols
        self.mapper = Mapper("qam", cfg.modulation_order)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        self.rg_mapper = ResourceGridMapper(self.rg)

        if self.cfg.precoding_method == "ZF":
            # the zero forcing precoder
            self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)
        elif self.cfg.precoding_method == "BD":
            self.bd_precoder = BDPrecoder(self.rg, sm, return_effective_channel=True)
            self.bd_equalizer = BDEqualizer(self.rg, sm)
        elif self.cfg.precoding_method == "SLNR":
            self.slnr_precoder = SLNRPrecoder(self.rg, sm, return_effective_channel=True)
            self.slnr_equalizer = SLNREqualizer(self.rg, sm)
        elif self.cfg.precoding_method == "ZF_QUANTIZED_CSI":
            self.zf_quantized_precoder = QuantizedZFPrecoder(self.rg, sm)
        elif self.cfg.precoding_method == "DIRECT_QUANTIZED_CSI":
            self.quantized_direct_precoder = QuantizedDirectPrecoder(self.rg, sm)
        else:
            ValueError(f"MU_MIMO __init__: unsupported precoding method {self.cfg.precoding_method}")

        # The LS channel estimator will provide channel estimates and error variances
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")
        interp = RBwiseLinearInterp(self.rg.pilot_pattern, rb_size=cfg.rb_size)
        self.ls_estimator_rb_wise = LSChannelEstimator(self.rg, interpolator=interp)

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        self.lmmse_equ = LMMSEEqualizer(self.rg, sm)

        # The demapper produces LLR for all coded bits
        self.demapper = Demapper("maxlog", "qam", cfg.modulation_order)

        # The decoder provides hard-decisions on the information bits
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)

    def call(self, dmimo_chans: dMIMOChannels, h_freq_csi, info_bits):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param h_freq_csi: CSI feedback for precoding
        :param info_bits: information bits
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        # LDPC encoder processing
        info_bits = tf.reshape(info_bits, [self.batch_size, 1, self.rg.num_streams_per_tx,
                                           self.num_codewords, self.encoder.k])
        c = self.encoder(info_bits)
        c = tf.reshape(c, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])

        # Interleaving for coded bits
        d = self.intlvr(c)

        # QAM mapping for the OFDM grid
        x = self.mapper(d)
        x_rg = self.rg_mapper(x)

        # apply precoding to OFDM grids
        if self.cfg.precoding_method == "ZF":
            x_precoded, g = self.zf_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            nvar = 5e-2  # TODO optimize value
            x_precoded, g = self.slnr_precoder([x_rg, h_freq_csi, nvar, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "ZF_QUANTIZED_CSI":
            x_precoded, g = self.zf_quantized_precoder(x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks)
        elif self.cfg.precoding_method == "DIRECT_QUANTIZED_CSI":
            x_precoded, g = self.quantized_direct_precoder(x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks)
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if np.any(np.not_equal(self.cfg.random_sto_vals, 0)):
            x_precoded = add_timing_offset(x_precoded, self.cfg.random_sto_vals)
        if np.any(np.not_equal(self.cfg.random_cfo_vals, 0)):
            x_precoded = add_frequency_offset(x_precoded, self.cfg.random_cfo_vals)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y, _ = dmimo_chans([x_precoded, self.cfg.first_slot_idx])

        # make proper shape
        # y = y[:, :, :self.num_rxs_ant, :, :]
        # y = tf.gather(y, tf.reshape(self.cfg.scheduled_rx_ue_indices, [-1]), axis=2)
        y = tf.reshape(y, (self.batch_size, self.num_rx_ue, self.num_ue_ant, 14, -1))

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            y = self.slnr_equalizer([y, h_freq_csi, nvar, self.cfg.ue_indices, self.cfg.ue_ranks])

        # LS channel estimation with linear interpolation
        no = 5e-2  # initial noise estimation (tunable param)
        if "DIRECT_QUANTIZED_CSI" in self.cfg.precoding_method:
            h_hat, err_var = self.ls_estimator_rb_wise([y, no])  # without interpolation
        else:
            h_hat, err_var = self.ls_estimator([y, no])
        # h_hat and h_hat2 have the shape of [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols ,num_effective_subcarriers]
        # if you reshape h_hat2 to [batch_size, num_rx, num_rx_ant, num_tx_ant, num_pilot_syms, num_effective_subcarriers],
        # you can see the channel estimates on each pilot OFDM symbol and each effective subcarrier, but in the subcarrier dimension,
        # only the pilot subcarriers have non-zero values. The pilots subcarriers are determined by self.rg.pilot_pattern
        # but in our code we are doing Kronecker pattern, so h_hat2[...,i,:,j] is non-zero only when j%num_tx_streams == 0
        # err_var2 has the shape of [1, 1, 1, 1, num_tx_ant, num_pilot_syms*num_effective_subcarriers]
        

        # # Debug: compare channel estimates
        # chan_perfect = tf.gather(_, self.rg.effective_subcarrier_ind, axis=-1)
        # chan_perfect = tf.transpose(chan_perfect, [0,1,3,5,6,2,4])

        # g = tf.gather(g, self.rg.effective_subcarrier_ind, axis=2)
        # g = g[tf.newaxis, tf.newaxis, ...]

        # h_eff_perfect = tf.matmul(chan_perfect, g)
        # # h_eff_perfect has shape of    [batch_size, 1, num_tx, num_ofdm_symbols, num_effective_subcarriers, num_rx*num_rx_ant, num_streams_per_tx]
        # # transpose to                  [batch_size, 1, num_rx*num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols ,num_effective_subcarriers]
        # h_eff_perfect = tf.transpose(h_eff_perfect, [0,1,5,2,6,3,4])
        # # Reshape to separate UE antennas
        # h_eff_perfect = tf.reshape(h_eff_perfect, (self.batch_size, self.num_rx_ue, self.num_ue_ant,
        #                                            self.rg.num_tx, self.num_streams_per_tx,
        #                                            self.rg.num_ofdm_symbols, self.rg.num_effective_subcarriers))
        # plt.figure(figsize=(10,6))
        # tx = 1
        # rx = 1
        # start_sc = 0
        # end_sc = (200//self.cfg.rb_size)*self.cfg.rb_size
        # plt.plot(np.real(h_eff_perfect[0,0,rx,0,tx,0,start_sc:end_sc]),'-*', label='perfect')
        # plt.plot(np.real(h_hat2[0,0,rx,0,tx,0,start_sc:end_sc]),'-*', label='estimated')
        # plt.plot(np.real(h_hat[0,0,rx,0,tx,0,start_sc:end_sc]),'-*', label='estimated new')
        # for i in range(0, end_sc, self.cfg.rb_size):
        #     plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
        # plt.legend()
        # plt.grid()
        # plt.savefig('a')

        # LMMSE equalization
        x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])

        # Soft-output QAM demapper
        llr = self.demapper([x_hat, no_eff])

        # Hard-decision bit error rate
        d_hard = tf.cast(llr > 0, tf.float32)
        uncoded_ber = compute_ber(d, d_hard).numpy()

        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x - x_hard) / np.prod(x.shape)
        node_wise_uncoded_ser = compute_UE_wise_SER(x ,x_hard, self.cfg.ue_ranks[0], self.cfg.num_tx_streams)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)

        sinr_dB_arr = None
        return dec_bits, uncoded_ber, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_dB_arr


def do_rank_link_adaptation(cfg, dmimo_chans, h_est, rx_snr_db):

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                        architecture='MU-MIMO', snrdb=rx_snr_db, fft_size=cfg.fft_size,
                                        precoder=cfg.precoding_method)
    if cfg.rank_adapt:
        rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')
    else:
        rank_feedback_report = [cfg.ue_ranks[0]]
    
    rank = rank_feedback_report[0]

    h_eff = rank_adaptation.calculate_effective_channel(rank, h_est)
    snr_linear = 10**(rx_snr_db/10)
    snr_linear = np.sum(snr_linear, axis=(2))
    snr_linear = np.mean(snr_linear)

    n_var = rank_adaptation.cal_n_var(h_eff, snr_linear)

    # Link adaptation
    if cfg.link_adapt:
        data_sym_position = np.arange(0, 14)
        link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                        architecture='MU-MIMO', snrdb=rx_snr_db, nfft=cfg.fft_size,
                                        N_s=rank, data_sym_position=data_sym_position, lookup_table_size='short')

        mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO')
    else:
        mcs_feedback_report = [[cfg.modulation_order], [cfg.code_rate]]

    return rank_feedback_report, n_var, mcs_feedback_report

3
def sim_mu_mimo(cfg: SimConfig, ns3cfg: Ns3Config, rc_config:RCConfig):
    """
    Simulation of MU-MIMO scenarios using different settings

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    :return: [uncoded_ber, coded_ber], [goodbits, userbits]
    """

    # CFO and STO settings
    if cfg.gen_sync_errors:
        cfg.random_sto_vals = cfg.sto_sigma * np.random.normal(size=(ns3cfg.num_txue_sel, 1))
        cfg.random_cfo_vals = cfg.cfo_sigma * np.random.normal(size=(ns3cfg.num_txue_sel, 1))

    # Reset UE selection. Start with all TX and RX UEs selected.
    # TODO: verify with Donald if this implementation is good
    # TODO: check if tx UEs are being handled properly downstream
    tmp_num_rxue_sel = ns3cfg.num_rxue_sel
    tmp_num_txue_sel = ns3cfg.num_txue_sel
    ns3cfg.reset_ue_selection()
    tx_ue_mask, rx_ue_mask = update_node_selection(cfg, ns3cfg)
    ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

    if not cfg.scheduling:
        rx_ue_mask = np.zeros(10)
        tx_ue_mask = np.zeros(10)
        rx_ue_mask[:tmp_num_rxue_sel] = 1
        tx_ue_mask[:tmp_num_txue_sel] = 1
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

        ue_indices = [[0, 1],[2, 3]] # Assuming gNB was scheduled
        scheduled_rx_UEs = np.arange(1, tmp_num_rxue_sel+1)
        for ue_idx in scheduled_rx_UEs:
            start = (ue_idx - 1) * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            end = ue_idx * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            ue_indices.append(list(np.arange(start, end)))
        cfg.scheduled_rx_ue_indices = np.array(ue_indices)
        cfg.num_scheduled_ues = cfg.scheduled_rx_ue_indices.shape[0]-2
        if not cfg.rank_adapt:
            cfg.num_tx_streams = (cfg.num_scheduled_ues+2) * cfg.ue_ranks[0]

        ue_indices = [[0, 1],[2, 3]] # Assuming gNB was scheduled
        scheduled_tx_UEs = np.arange(1, tmp_num_txue_sel+1)
        for ue_idx in scheduled_tx_UEs:
            start = (ue_idx - 1) * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            end = ue_idx * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            ue_indices.append(list(np.arange(start, end)))
        cfg.scheduled_tx_ue_indices = np.array(ue_indices)

    # dMIMO channels from ns-3 simulator
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
    
    # Total number of antennas in the TxSquad, always use all gNB antennas
    num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant

    # Adjust guard subcarriers for channel estimation grid
    csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
    csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
    csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

    # Resource grid for channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_txs_ant,
                          cyclic_prefix_length=cfg.cyclic_prefix_len,
                          num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    # Channel CSI estimation using channels in previous frames/slots
    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db, rx_pwr_dbm = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx,
                                                                     batch_size=cfg.num_slots_p2)
    elif cfg.csi_prediction is True:
        rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams, ns3cfg)
        # Get CSI history
        # TODO: optimize channel estimation and optimization procedures (currently very slow)
        # TODO: Add graph ESN implementation here as an option
        

        start_time = time.time()
        if cfg.use_perfect_csi_history_for_prediction:
            h_freq_csi_history = rc_predictor.get_ideal_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                          dmimo_chans)
        else:        
            h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                            rg_csi, dmimo_chans, 
                                                            cfo_vals=cfg.random_cfo_vals,
                                                            sto_vals=cfg.random_sto_vals,
                                                            estimated_channels_dir=cfg.estimated_channels_dir)
        end_time = time.time()
        print("Total time for channel history gathering: ", end_time - start_time)
        
        if cfg.channel_prediction_method == "two_mode":
            start_time = time.time()
            T, _, _, RxAnt, _, TxAnt, num_syms, RB = h_freq_csi_history.shape
            h_freq_csi = np.zeros(h_freq_csi_history[0,...].shape, dtype=h_freq_csi_history.dtype)
            num_bs_ant = 4
            num_ue_ant = 2

            for tx_node_idx in range(ns3cfg.num_txue_sel+1):
                for rx_node_idx in range(ns3cfg.num_rxue_sel+1):
                    if tx_node_idx == 0:
                        tx_ant_idx = np.arange(0,num_bs_ant)
                    else:
                        tx_ant_idx = np.arange(num_bs_ant + (tx_node_idx-1)*num_ue_ant,num_bs_ant + (tx_node_idx)*num_ue_ant)
                    TxAnt = len(tx_ant_idx)

                    if rx_node_idx == 0:
                        rx_ant_idx = np.arange(0,num_bs_ant)
                    else:
                        rx_ant_idx = np.arange(num_bs_ant + (rx_node_idx-1)*num_ue_ant,num_bs_ant + (rx_node_idx)*num_ue_ant)
                    RxAnt = len(rx_ant_idx)

                    curr_h_freq_csi_history = h_freq_csi_history[:,:,:,rx_ant_idx,:,...]
                    curr_h_freq_csi_history = curr_h_freq_csi_history[:,:,:,:,:,tx_ant_idx,...]
                    
                    twomode_predictor = twomode_wesn_pred(rc_config=rc_config, 
                                                num_freq_re=RB, 
                                                num_rx_ant=RxAnt, 
                                                num_tx_ant=TxAnt
                                                )
                    rx_idx, tx_idx = np.ix_(rx_ant_idx, tx_ant_idx)
                    tmp = np.asarray(twomode_predictor.predict(curr_h_freq_csi_history))
                    h_freq_csi[:, :, rx_idx, :, tx_idx, :, :] = tmp.transpose(2, 4, 0, 1, 3, 5, 6)
            end_time = time.time()
            print("total time for training and prediction: ", end_time-start_time)
        elif cfg.channel_prediction_method == "old":
            h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
        else:
            raise ValueError("Channel prediction method not implemented here.")
    else:
        # LMMSE channel estimation. h_freq_csi shape: [_, _, num_rx_ants, _ num_tx_ants, num_syms, num_subcarriers]
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals)
    _, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                batch_size=cfg.num_slots_p2)

    ###############################################################################################################################################
    # Testing scheduler without explicit CSI calculation for now. Using right singular vectors and singular values as a surrogate for CSI.
    ###############################################################################################################################################

    if cfg.scheduling: # Different from UE selection. Now, you treat the number of RX Squad UEs from UE selection as the maximum number to schedule. You schedule somewhere between 1 to ns3cfg.num_rxue_sel UEs in the RX Squad
        # TODO: Currently only exhaustive search is implemented fully
        # TODO: Currently, we use only 1 (outdated) CSI feedback to do RX UE selection. Should base it on some long-term statistics to reduce the impact of feedback delay

        # Get right singular vectors and singular values of effective channels 
        num_rx_nodes = ns3cfg.num_rxue_sel+1
        h_reconstructed = np.zeros((h_freq_csi.shape), dtype=complex)
        h_reconstructed = h_reconstructed[:, :, :(num_rx_nodes+1)*cfg.ue_ranks[0], ...]
        for rx_node_idx in range(num_rx_nodes):

            if rx_node_idx == 0:
                ant_idx = np.arange(4)
                num_streams = 2
                stream_idx = np.arange(2)
            else:
                ant_idx = np.arange((rx_node_idx-1) * 2 + 4, rx_node_idx * 2 + 4)
                num_streams = 1
                stream_idx = np.arange((rx_node_idx-1) * cfg.ue_ranks[0] + cfg.ue_ranks[0]*2, rx_node_idx * cfg.ue_ranks[0] + cfg.ue_ranks[0]*2)

            curr_h = tf.gather(h_freq_csi, ant_idx, axis=2)

            s, _, Vh = tf.linalg.svd(curr_h, full_matrices=False)
            s = tf.cast(s, Vh.dtype)
            s = tf.expand_dims(s, axis=-2)

            curr_h_reconstructed = Vh * s
            curr_h_reconstructed = curr_h_reconstructed[:,:, :num_streams, ...]
            # curr_h_reconstructed = curr_h_reconstructed[:,:,:,tf.newaxis,...]

            h_reconstructed[:, :, stream_idx, ...] = tf.transpose(curr_h_reconstructed, [0, 1, 2, 3, 4, 6, 5])
        
        h_reconstructed = tf.convert_to_tensor(h_reconstructed)

        # Scheduling
        mu_mimo_scheduler = MUMIMOScheduler(rx_snr_db, max_rx_UEs_scheduled=tmp_num_rxue_sel)
        scheduled_rx_nodes = mu_mimo_scheduler(h_reconstructed)
        scheduled_rx_UEs = scheduled_rx_nodes[1:] - 1 # Assuming gNB was scheduled
        
        # Updating system parameters based on scheduling    
        rx_ue_mask = np.zeros(10)
        tx_ue_mask = np.zeros(10)
        rx_ue_mask[scheduled_rx_UEs] = 1
        tx_ue_mask[:tmp_num_txue_sel] = 1
        ns3cfg.update_ue_selection(tx_ue_mask, rx_ue_mask)

        ue_indices = [[0, 1],[2, 3]] # Assuming gNB was scheduled
        for node in scheduled_rx_nodes[1:]:
            start = (node - 1) * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            end = node * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            ue_indices.append(list(np.arange(start, end)))
        cfg.scheduled_rx_ue_indices = np.array(ue_indices)
        cfg.num_scheduled_ues = scheduled_rx_UEs.size
        cfg.num_tx_streams = (cfg.num_scheduled_ues+2) * cfg.ue_ranks[0]

        # TODO: Currently, TX UEs not selected intelligently (just select the first tmp_num_txue_sel UEs)
        ue_indices = [[0, 1],[2, 3]] # Assuming gNB was scheduled
        scheduled_tx_UEs = np.arange(1, tmp_num_txue_sel+1)
        for ue_idx in scheduled_tx_UEs:
            start = (ue_idx - 1) * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            end = ue_idx * ns3cfg.num_ue_ant + ns3cfg.num_bs_ant
            ue_indices.append(list(np.arange(start, end)))
        cfg.scheduled_tx_ue_indices = np.array(ue_indices)
    
    # Pick the selected UE's channels
    h_freq_csi = tf.gather(h_freq_csi, tf.reshape(cfg.scheduled_rx_ue_indices, (-1,)), axis=2)
    h_freq_csi = tf.gather(h_freq_csi, tf.reshape(cfg.scheduled_tx_ue_indices, (-1,)), axis=4)

    # Rank and link adaptation
    rank_feedback_report, n_var, mcs_feedback_report = \
        do_rank_link_adaptation(cfg, dmimo_chans, h_freq_csi, rx_snr_db)

    if "QUANTIZED_CSI" in cfg.precoding_method:
        num_tx_ant = h_freq_csi.shape[4]
        donald_hack = True
        quantization_debug = False
        rvq = RandomVectorQuantizer(bits_per_codeword=15, vector_dim=num_tx_ant, seed=42)
        # rvq = RandomVectorQuantizerNumpy(bits_per_codeword=15, vector_dim=h_freq_csi.shape[4], seed=42)
        h_freq_per_rx = []
        # Adjust guard subcarriers for different number of streams
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
            h_freq_rx = h_freq_csi[:, :, i_rxnode*2:(i_rxnode+1)*2, : , :, :, guard_carriers_1:-guard_carriers_2] 
            # h_freq_rx shape: [batch_size, num_rx=1, num_rx_ants=2, num_tx=1, num_tx_ants, num_syms, num_effective_subcarriers]
            # transpose to [batch_size, num_rx=1, num_tx=1, num_syms, num_subcarriers, num_rx_ants=2, num_tx_ants]
            H = tf.transpose(h_freq_rx, perm=[0, 1, 3, 5, 6, 2, 4])
            
            num_syms = H.shape[3]
            H = tf.reduce_mean(H, axis=3, keepdims=True) # average over num_syms dimension # [B, num_rx=1, num_tx=1, 1, num_subcarriers, num_rx_ants=2, num_tx_ants]
            n_sc = H.shape[4]; B = H.shape[0]
            num_rbs = n_sc // cfg.rb_size
            # if n_sc % cfg.rb_size != 0:
            #     raise ValueError(f"Number of subcarriers for CSI feedback ({n_sc}) is not a multiple of rb_size ({cfg.rb_size})")
            # Reshape and average over RBs
            # Here we have a problem where n_sc may not be multiple of rb_size due to guard bands
            # We will fix that by including the last residual subcarriers into the last RB
            if n_sc % cfg.rb_size == 0:
                H = tf.reshape(H, [B , 1, 1, 1, n_sc//cfg.rb_size, cfg.rb_size, 2, num_tx_ant]) # Reshape to RBGs # [B, num_rx=1, num_tx=1, 1, num_rb, cfg.rb_size, 2, num_tx_ants]
                num_residual_subcarriers = 0
                H = tf.reduce_mean(H, axis=5, keepdims=True) # Average over each RBG # [B, num_rx=1, num_tx=1, 1, num_rb, 1, 2, num_tx_ants]
                n_sc_less_residual = n_sc
            else:
                num_residual_subcarriers = n_sc % cfg.rb_size
                n_sc_less_residual = n_sc - (num_residual_subcarriers)
                H_less_last_rb = H[:, :, :, :, :-(cfg.rb_size + num_residual_subcarriers)] # [B, num_rx=1, num_tx=1, 1, n_sc - (cfg.rb_size + num_residual_subcarriers), num_rx_ants=2, num_tx_ants]
                H_last_rb = H[:, :, :, :, -(cfg.rb_size + num_residual_subcarriers):] # [B, num_rx=1, num_tx=1, 1, cfg.rb_size + num_residual_subcarriers, num_rx_ants=2, num_tx_ants]
                # Reshape to RBGs # [B, num_rx=1, num_tx=1, 1, num_rbs - 1, cfg.rb_size, 2, num_tx_ants]
                H_less_last_rb = tf.reshape(H_less_last_rb, [B , 1, 1, 1, num_rbs - 1, cfg.rb_size, 2, num_tx_ant]) 
                # Reshape last RB # [B, num_rx=1, num_tx=1, 1, 1, cfg.rb_size + num_residual_subcarriers, 2, num_tx_ants]
                H_last_rb = tf.reshape(H_last_rb, [B , 1, 1, 1, 1, cfg.rb_size + num_residual_subcarriers, 2, num_tx_ant])
                H_less_last_rb = tf.reduce_mean(H_less_last_rb, axis=5, keepdims=True) # Average over each RB # [B, num_rx=1, num_tx=1, 1, num_rbs - 1, 1, 2, num_tx_ants]
                H_last_rb = tf.reduce_mean(H_last_rb, axis=5, keepdims=True) # Average over last RB # [B, num_rx=1, num_tx=1, 1, 1, 1, 2, num_tx_ants]
                H = tf.concat([H_less_last_rb, H_last_rb], axis=4) # concatenate back # [B, num_rx=1, num_tx=1, 1, num_rb, 1, 2, num_tx_ants]
            # Now H is of shape [B, num_rx=1, num_tx=1, 1, num_rbs, 1, 2, num_tx_ants]
            if donald_hack and ("DIRECT" not in cfg.precoding_method):
                H_avg = tf.reduce_mean(H, axis=-2) # Average over num_rx_ants dimension (the Donald hack) # [B, num_rx=1, num_tx=1, 1, num_rb, 1, num_tx_ants]
                H_avg_norm = (tf.linalg.norm(H_avg, axis=-1, keepdims=True) + 1e-12) # [B, num_rx=1, num_tx=1, 1, num_rb, 1, 1]
                H_avg_normalized = H_avg / H_avg_norm  # Normalize over num_tx_ants dimension (each vector is norm 1 now) # [B, num_rx=1, num_tx=1, 1, num_rb, 1, num_tx_ants]
                H_avg_normalized_quantized = rvq(H_avg_normalized) # shape : [B, num_rx=1, num_tx=1, 1, num_rb, 1] dtype=int
                H_avg_normalized_reconstructed = rvq(H_avg_normalized_quantized) # shape : [B, num_rx=1, num_tx=1, 1, num_rb, 1, num_tx_ants]
                H_avg_reconstructed = (H_avg_normalized_reconstructed)# * H_avg_norm) # Scale back to original norm
                # H_avg_reconstructed = H_avg_normalized * H_avg_norm # Debugging: skip quantization
                H_avg_reconstructed = tf.tile(H_avg_reconstructed, [1, 1, 1, num_syms, 1, cfg.rb_size, 1]) # [B, num_rx=1, num_tx=1, num_syms, num_rb, cfg.rb_size, num_tx_ant]
                H_avg_reconstructed = tf.reshape(H_avg_reconstructed, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])  # [B, num_syms, num_effective_subarriers, 1, num_tx_ants]
                # here we have a problem where n_sc may not be multiple of rb_size due to guard bands
                # We will fix that by repeating the last residual subcarriers into the H_avg_reconstructed
                if num_residual_subcarriers != 0:
                    H_avg_reconstructed = tf.concat([
                        H_avg_reconstructed,
                        H_avg_reconstructed[:, :, -(num_residual_subcarriers):, :, :],
                    ], axis=2)
                if quantization_debug:
                    # print the distortion introduced by quantization
                    H_avg1 = tf.tile(H_avg, [1, 1, 1, num_syms, 1, cfg.rb_size, 1]) # [B, num_rx=1, num_tx=1, num_syms, num_rb, cfg.rb_size, num_tx_ant]
                    H_avg1 = tf.reshape(H_avg1, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])  # [B, num_syms, num_effective_subarriers, 1, num_tx_ants]
                    print(f"For RX node {i_rxnode}, quantization distortion (Frobenius norm) norm(actual - reconstructed) / norm(actual):" + 
                        f" {tf.linalg.norm(H_avg1 - H_avg_reconstructed) / tf.linalg.norm(H_avg1)}")
                h_freq_per_rx.append(tf.transpose(H_avg_reconstructed, perm=[0, 3, 4, 1, 2])) # transpose to [batch_size, num_streams=1, num_tx_ants , num_ofdm_symbols, num_effective_subarriers]
            else:
                s, u , v = tf.linalg.svd(H)
                v_largest = v[..., 0]                     # [B, num_rx=1, num_tx=1, 1, num_rb, 1, 2]
                # repeat to original num_syms and subcarrier dimension
                v_largest = tf.tile(v_largest, [1, 1, 1, num_syms, 1, cfg.rb_size, 1])  # [B, num_rx=1, num_tx=1, num_syms, num_rb, cfg.rb_size, num_tx_ant]
                v_largest = tf.reshape(v_largest, [B, num_syms, n_sc_less_residual, 1, num_tx_ant])  # [B, num_syms, num_effective_subarriers, 1, num_tx_ants]
                # here we have a problem where n_sc may not be multiple of rb_size due to guard bands
                # We will fix that by repeating the last residual subcarriers into the v_largest
                if num_residual_subcarriers != 0:
                    v_largest = tf.concat([
                        v_largest,
                        v_largest[:, :, -(num_residual_subcarriers):, :, :],
                    ], axis=2)
                v_largest_quantized = rvq(v_largest) # shape : [B, num_syms, num_effective_subarriers, 1] dtype=int
                v_largest_reconstructed = rvq(v_largest_quantized) # shape : [B, num_syms, num_effective_subarriers, 1, num_tx_ants]
                vh = tf.linalg.adjoint(v_largest_reconstructed)  # [B, num_syms, num_effective_subarriers, num_tx_ants, num_streams=1]
                h_freq_per_rx.append(tf.transpose(vh, perm=[0, 4, 3, 1, 2])) # transpose to [batch_size, num_streams=1, num_tx_ants , num_ofdm_symbols, num_effective_subarriers]
        h_freq_quantized = tf.concat(h_freq_per_rx, axis=1) # concatenate along the num_streams dimension
        # final h_freq_quantized shape: [batch_size, num_streams, num_tx_ants, num_ofdm_symbols, num_effective_subarriers]
        # we need to change the shape to [batch_size, num_streams, num_tx_ants, num_ofdm_symbols, fftsize]
        # by repeating the first and last element of final_h_freq_quantized along the last dimension as
        # much as the number of guard carriers 
        first_subcarrier = tf.repeat(h_freq_quantized[..., 0:1], repeats=guard_carriers_1, axis=-1)
        last_subcarrier = tf.repeat(h_freq_quantized[..., -1:], repeats=guard_carriers_2, axis=-1)
        h_freq_csi = tf.concat([first_subcarrier, h_freq_quantized, last_subcarrier], axis=-1)
        

    if cfg.rank_adapt:
        # Update rank and total number of streams
        rank = rank_feedback_report[0]
        cfg.ue_ranks = [rank]
        cfg.num_tx_streams = rank * (cfg.num_scheduled_ues + 2)  # treat BS as two UEs

        # print("\n", "rank per user (MU-MIMO) = ", rank, "\n")
        # print("\n", "rate per user (MU-MIMO) = ", rate, "\n")

    if cfg.link_adapt:
        
        qam_order_arr = mcs_feedback_report[0]
        code_rate_arr = mcs_feedback_report[1]
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.modulation_order = int(most_frequent_value)

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.code_rate = most_frequent_value

        # print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order, "\n")
        # print("\n", "Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")

    ranks_out = int(cfg.num_tx_streams / (cfg.num_scheduled_ues+2))

    # Create MU-MIMO simulation
    mu_mimo = MU_MIMO(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    # TxSquad transmission (P1)
    if cfg.enable_txsquad is True:
        tx_squad = TxSquad(cfg, ns3cfg, mu_mimo.num_bits_per_frame)
        txs_chans = dMIMOChannels(ns3cfg, "TxSquad", add_noise=True)
        info_bits_new, txs_ber, txs_bler = tx_squad(txs_chans, info_bits)
        # print("BER: {}  BLER: {}".format(txs_ber, txs_bler))
        assert txs_ber <= 1e-3, "TxSquad transmission BER too high"

    # MU-MIMO transmission (P2)
    dec_bits, uncoded_ber_phase_2, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_dB_arr = mu_mimo(dmimo_chans, h_freq_csi, info_bits)

    # Update error statistics
    info_bits = tf.reshape(info_bits, dec_bits.shape) # shape: [batch_size, 1, num_streams_per_tx, num_codewords, num_effective_subcarriers*num_data_ofdm_syms_per_subframe]
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()

    node_wise_ber, node_wise_bler = compute_UE_wise_BER(info_bits, dec_bits, cfg.ue_ranks[0], cfg.num_tx_streams)

    # RxSquad transmission (P3)
    if cfg.enable_rxsquad is True:
        rxcfg = cfg.clone()
        rxcfg.csi_delay = 4
        rxcfg.decoder = "lmmse"
        rxcfg.perfect_csi = False
        rxcfg.first_slot_idx = cfg.first_slot_idx + cfg.num_slots_p2
        num_ue_bits_per_frame = mu_mimo.num_bits_per_frame * (cfg.num_scheduled_ues / (cfg.num_scheduled_ues + 2))

        rx_ns3cfg = Ns3Config(data_folder=cfg.ns3_folder, total_slots=cfg.total_slots)
        rx_ns3cfg.update_ue_selection(None, rx_ue_mask)
        rxs_chans = dMIMOChannels(rx_ns3cfg, "RxSquad", add_noise=False)
        rx_squad = RxSquad(rxcfg, ns3cfg, num_ue_bits_per_frame, rxs_chans)
        print("Each RxSquad UE transmitting {} streams, each with modulation order {}".format(rx_squad.num_streams_per_tx, rx_squad.num_bits_per_symbol_per_UE))

        forwarding_bits = dec_bits[:,:,-(cfg.num_scheduled_ues * cfg.ue_ranks[0]):, : , :]
        dec_bits_phase_3, \
        node_wise_uncoded_ber_phase_3, \
        uncoded_ber_phase_3, \
        node_wise_coded_ber_phase_3, \
        coded_ber_phase_3, \
        node_wise_coded_bler_phase_3, \
        coded_bler_phase_3 = rx_squad(rxs_chans, forwarding_bits)
        # print("PHASE 3 STATS\nUNCODED BER: {}\nCODED BER: {}\nBLER: {}".format(uncoded_ber_phase_3 , coded_ber_phase_3, coded_bler_phase_3))
        # if uncoded_ber_phase_3 >= 1e-2 or coded_ber_phase_3 >= 1e-2:
        #     print("Warning: High RxSquad transmission BER")
        
        dec_bits_phase_3 = tf.reshape(dec_bits_phase_3, [dec_bits_phase_3.shape[0], forwarding_bits.shape[0], forwarding_bits.shape[1], forwarding_bits.shape[3], forwarding_bits.shape[4]])
        dec_bits_phase_3 = tf.transpose(dec_bits_phase_3, perm=[1, 2, 0, 3, 4])
        gNB_bits_phase_2 = dec_bits[:,:,:-(cfg.num_scheduled_ues * cfg.ue_ranks[0]), : , :]
        end_to_end_dec_bits = tf.concat([gNB_bits_phase_2, dec_bits_phase_3], axis=2)

        coded_ber = compute_ber(info_bits, end_to_end_dec_bits).numpy()
        coded_bler = compute_bler(info_bits, end_to_end_dec_bits).numpy()

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * mu_mimo.num_bits_per_frame
    userbits = (1.0 - coded_bler) * mu_mimo.num_bits_per_frame
    ratedbits_phase_2 = (1.0 - uncoded_ser) * mu_mimo.num_uncoded_bits_per_frame

    node_wise_goodbits_phase_2 = (1.0 - node_wise_ber) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_userbits_phase_2 = (1.0 - node_wise_bler) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_ratedbits_phase_2 = (1.0 - node_wise_uncoded_ser) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)

    return [uncoded_ber_phase_2, coded_ber], [goodbits, userbits, ratedbits_phase_2], [node_wise_goodbits_phase_2, node_wise_userbits_phase_2, node_wise_ratedbits_phase_2, ranks_out, sinr_dB_arr]


def sim_mu_mimo_all(cfg: SimConfig, ns3cfg: Ns3Config, rc_config:RCConfig):
    """"
    Simulation of MU-MIMO scenario according to the frame structure

    :param cfg: simulation settings
    :param ns3cfg: ns-3 channel settings
    """

    slot_time = cfg.slot_duration  # default 1ms subframe/slot duration
    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)

    total_cycles = 0
    uncoded_ber, ldpc_ber, goodput, throughput, bitrate = 0, 0, 0, 0, 0
    nodewise_goodput = []
    nodewise_throughput = []
    nodewise_bitrate = []
    ranks_list = []
    ldpc_ber_list = []
    uncoded_ber_list = []
    sinr_dB_list = []
    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        
        print("first_slot_idx: ", first_slot_idx, "\n")

        total_cycles += 1
        cfg.first_slot_idx = first_slot_idx

        start_time = time.time()
        bers, bits, additional_KPIs = sim_mu_mimo(cfg, ns3cfg, rc_config)
        end_time = time.time()
        print("Cycle time: ", end_time - start_time, " seconds\n")
        
        uncoded_ber += bers[0]
        ldpc_ber += bers[1]
        uncoded_ber_list.append(bers[0])
        ldpc_ber_list.append(bers[1])
        
        goodput += bits[0]
        throughput += bits[1]
        bitrate += bits[2]
        
        nodewise_goodput.append(additional_KPIs[0])
        nodewise_throughput.append(additional_KPIs[1])
        nodewise_bitrate.append(additional_KPIs[2])
        ranks_list.append(additional_KPIs[3])
        sinr_dB_list.append(additional_KPIs[4])

    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    bitrate = bitrate / (total_cycles * slot_time * 1e6) * overhead  # Mbps

    nodewise_goodput = np.concatenate(nodewise_goodput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_throughput = np.concatenate(nodewise_throughput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_bitrate = np.concatenate(nodewise_bitrate) / (slot_time * 1e6) * overhead  # Mbps
    ranks = np.array(ranks_list).flatten()
    if sinr_dB_list[0] is not None:
        sinr_dB = np.concatenate(sinr_dB_list)
    else:
        sinr_dB = None

    return [uncoded_ber/total_cycles, ldpc_ber/total_cycles, goodput, throughput, bitrate, nodewise_goodput, nodewise_throughput, nodewise_bitrate, ranks, uncoded_ber_list, ldpc_ber_list, sinr_dB]

