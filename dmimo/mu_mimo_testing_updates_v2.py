import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
import matplotlib.pyplot as plt
import time
from typing import Optional
from pathlib import Path

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.interleaving import RowColumnInterleaver, Deinterleaver

from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, flatten_dims, matrix_inv, matrix_pinv
from sionna.utils.metrics import compute_ber, compute_bler

from dmimo.config import Ns3Config, SimConfig, RCConfig
from dmimo.channel import (
    dMIMOChannels,
    lmmse_channel_estimation,
    estimate_freq_cov,
    LMMSELinearInterp,
    estimate_channel_from_pilot_rx_symbols,
)
from dmimo.channel.wesn_rx_sig_pred import wesn_rx_sig_pred
from dmimo.channel import standard_rc_pred_freq_mimo, default_ddpg_predictor
from dmimo.channel.ddpg_predictor import DDPGChannelPredictor
from dmimo.channel import twomode_wesn_pred, twomode_wesn_pred_tf, weiner_filter_pred, kalman_filter_pred
from dmimo.channel.twomode_wesn_pred import predict_all_links, predict_all_links_simple
from dmimo.channel.configured_wesn_pred import (
    build_configured_predictors_simple,
    predict_all_links_with_configured_simple,
)
from dmimo.channel.wesn_rx_sig_pred import rx_sig_predict_all_links_simple
from dmimo.channel.rl_beam_selector_v2 import RLBeamSelector
from dmimo.channel.twomode_wesn_pred_tf import predict_all_links_tf
from dmimo.channel import RBwiseLinearInterp
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, QuantizedSLNRPrecoder, SLNREqualizer, QuantizedZFPrecoder, QuantizedDirectPrecoder
from dmimo.mimo import rankAdaptation, linkAdaptation
from dmimo.mimo import MUMIMOScheduler
from dmimo.mimo import update_node_selection, quantized_CSI_feedback, RandomVectorQuantizer, RandomVectorQuantizerNumpy
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER, complex_pinv

from .txs_mimo import TxSquad
from .rxs_mimo import RxSquad

def _compute_effective_snr_db(h_freq_csi, err_var_csi, eps=1e-12):
    """Compute effective post-estimation SNR in dB from channel and error variance."""

    if h_freq_csi is None or err_var_csi is None:
        return None

    h_np = np.asarray(h_freq_csi)
    err_np = np.asarray(err_var_csi)
    eff_snr_lin = np.mean(np.abs(h_np) ** 2 / np.maximum(err_np, eps))
    return 10.0 * np.log10(eff_snr_lin + eps)


def _print_lmmse_effective_snr(tag, h_freq_csi, err_var_csi):
    """Print effective SNR after LMMSE channel estimation."""

    snr_db = _compute_effective_snr_db(h_freq_csi, err_var_csi)
    if snr_db is None:
        return
    print(f"{tag} Effective SNR after LMMSE channel estimation: {snr_db:.2f} dB")

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

        if "ZF" in self.cfg.precoding_method:
            self.zf_precoder = ZFPrecoder(self.rg, sm, return_effective_channel=True)
            self.zf_quantized_precoder = QuantizedZFPrecoder(self.rg, sm)
        elif self.cfg.precoding_method == "BD":
            self.bd_precoder = BDPrecoder(self.rg, sm, return_effective_channel=True)
            self.bd_equalizer = BDEqualizer(self.rg, sm)
        elif self.cfg.precoding_method == "SLNR":
            self.slnr_precoder = SLNRPrecoder(self.rg, sm, return_effective_channel=True)
            self.slnr_equalizer = SLNREqualizer(self.rg, sm)
            self.slnr_quantized_precoder = QuantizedSLNRPrecoder(self.rg, sm)
            # self.slnr_quantized_equalizer = QuantizedSLNREqualizer(self.rg, sm)
        elif "DIRECT" in self.cfg.precoding_method:
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

    def call(self, dmimo_chans: dMIMOChannels, h_freq_csi, info_bits, snr_dB_arr):
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

        # apply precoding to OFDM grids. We currently assume either perfect CSI or quantized CSI feedback.
        if self.cfg.precoding_method == "ZF":
            if "RVQ" in self.cfg.PMI_feedback_architecture and self.cfg.csi_quantization_on:
                x_precoded, g = self.zf_quantized_precoder(x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks)
            elif "type_II" in self.cfg.PMI_feedback_architecture and self.cfg.csi_quantization_on:
                x_precoded, g = self.zf_quantized_precoder(x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks, new=True)
            else:
                x_precoded, g = self.zf_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "BD":
            x_precoded, g = self.bd_precoder([x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "SLNR":
            nvar = 5e-2  # TODO optimize value
            if "type_II" in self.cfg.PMI_feedback_architecture and self.cfg.csi_quantization_on:
                x_precoded, g = self.slnr_quantized_precoder(x_rg, h_freq_csi, snr_dB_arr, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks)
            else:
                x_precoded, g = self.slnr_precoder([x_rg, h_freq_csi, nvar, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks])
        elif self.cfg.precoding_method == "DIRECT":
            x_precoded, g = self.quantized_direct_precoder(x_rg, h_freq_csi, self.cfg.scheduled_rx_ue_indices, self.cfg.ue_ranks)
        else:
            ValueError("unsupported precoding method")

        # add CFO/STO to simulate synchronization errors
        if np.any(np.not_equal(self.cfg.random_sto_vals, 0)):
            x_precoded = add_timing_offset(x_precoded, self.cfg.random_sto_vals)
        if np.any(np.not_equal(self.cfg.random_cfo_vals, 0)):
            x_precoded = add_frequency_offset(x_precoded, self.cfg.random_cfo_vals)

        # apply dMIMO channels to the resource grid in the frequency domain.
        y, h_freq_true = dmimo_chans([x_precoded, self.cfg.first_slot_idx])
        # make proper shape
        # y = y[:, :, :self.num_rxs_ant, :, :]
        # y = tf.gather(y, tf.reshape(self.cfg.scheduled_rx_ue_indices, [-1]), axis=2)
        y = tf.reshape(y, (self.batch_size, self.num_rx_ue, self.num_ue_ant, 14, -1))

        if self.cfg.precoding_method == "BD":
            y = self.bd_equalizer([y, h_freq_csi, self.cfg.ue_indices, self.cfg.ue_ranks])
        # elif self.cfg.precoding_method == "SLNR":
        #     y = self.slnr_equalizer([y, h_freq_csi, nvar, self.cfg.ue_indices, self.cfg.ue_ranks])

        # LS channel estimation with linear interpolation
        no = 5e-2  # initial noise estimation (tunable param)
        if "DIRECT_QUANTIZED_CSI_RVQ" in self.cfg.precoding_method or "phase2" in self.cfg.PMI_feedback_architecture:
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
        # print(f"\nUncoded BER: {uncoded_ber}\n")

        nodewise_uncoded_ber, _ = compute_UE_wise_BER(d, d_hard, self.cfg.ue_ranks[0], self.cfg.num_tx_streams)

        # Hard-decision symbol error rate
        x_hard = self.mapper(d_hard)
        uncoded_ser = np.count_nonzero(x - x_hard) / np.prod(x.shape)
        node_wise_uncoded_ser = compute_UE_wise_SER(x ,x_hard, self.cfg.ue_ranks[0], self.cfg.num_tx_streams)

        # LLR deinterleaver for LDPC decoding
        llr = self.dintlvr(llr)
        llr = tf.reshape(llr, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords, self.encoder.n])

        # LDPC hard-decision decoding
        dec_bits = self.decoder(llr)

        sinr_linear = tf.math.reciprocal(tf.cast(no_eff, tf.float32) + 1e-12)
        # sinr_linear = tf.reduce_mean(sinr_linear, axis=[1, -1])
        # sinr_linear = tf.reduce_mean(sinr_linear, axis=0)
        sinr_dB_arr = 10 * np.log10(sinr_linear.numpy() + 1e-12)

        if bool(getattr(self.cfg, "return_first_slot_only", False)):
            d_first = d[0:1, ...]
            d_hard_first = d_hard[0:1, ...]
            x_first = x[0:1, ...]
            x_hard_first = x_hard[0:1, ...]

            dec_bits = dec_bits[0:1, ...]
            x_hat = x_hat[0:1, ...]
            uncoded_ber = compute_ber(d_first, d_hard_first).numpy()
            uncoded_ser = np.count_nonzero(x_first - x_hard_first) / np.prod(x_first.shape)
            node_wise_uncoded_ser = compute_UE_wise_SER(
                x_first,
                x_hard_first,
                self.cfg.ue_ranks[0],
                self.cfg.num_tx_streams,
            )
            if np.ndim(sinr_dB_arr) > 0:
                sinr_dB_arr = sinr_dB_arr[0:1, ...]

        return dec_bits, uncoded_ber, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_dB_arr


def do_rank_link_adaptation(cfg, dmimo_chans, h_est, rx_sinr_db, return_mcs_index=False):

    # Rank adaptation
    rank_adaptation = rankAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                        architecture='MU-MIMO', sinrdb=rx_sinr_db, fft_size=cfg.fft_size,
                                        precoder=cfg.precoding_method)
    if cfg.rank_adapt:
        rank_feedback_report = rank_adaptation(h_est, channel_type='dMIMO')
    else:
        rank_feedback_report = [cfg.ue_ranks[0]]
    
    rank = rank_feedback_report[0]

    # Link adaptation
    if cfg.link_adapt:
        data_sym_position = np.arange(0, 14)
        link_adaptation = linkAdaptation(dmimo_chans.ns3_config.num_bs_ant, dmimo_chans.ns3_config.num_ue_ant,
                                        architecture='MU-MIMO', sinrdb=rx_sinr_db, nfft=cfg.fft_size,
                                        N_s=rank, data_sym_position=data_sym_position, lookup_table_size='long')

        mcs_feedback_report = link_adaptation(h_est, channel_type='dMIMO', return_mcs_index=return_mcs_index)
    else:
        mcs_feedback_report = [[cfg.modulation_order], [cfg.code_rate], [None], [None]]

    return rank_feedback_report, mcs_feedback_report


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
    else:
        raise Exception ("Scheduling not supported in this version.")

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
    
    # Cacheable LMMSE resources for the current drop
    freq_cov_mat = getattr(cfg, "freq_cov_mat", None)
    lmmse_interpolator = getattr(cfg, "lmmse_interpolator", None)
    lmmse_use_rx_snr_for_nvar = getattr(cfg, "lmmse_use_rx_snr_for_nvar", True)

    # Channel CSI estimation using channels in previous frames/slots
    h_freq_csi_history = None
    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db, rx_pwr_dbm = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx,
                                                                     batch_size=cfg.num_slots_p2)
        h_freq_csi_perfect = h_freq_csi
    elif cfg.csi_prediction is True:
        rc_predictor = getattr(cfg, "rc_predictor", None)

        if rc_predictor is None:
            rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams, rc_config, ns3cfg)
            cfg.rc_predictor = rc_predictor
        if cfg.first_slot_idx == cfg.start_slot_idx:
            rc_predictor.reset_csi_history()
        
        # Get CSI history
        start_time = time.time()
        err_var_csi_history = None
        if cfg.use_perfect_csi_history_for_prediction:
            h_freq_csi_history = rc_predictor.get_ideal_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                          dmimo_chans)
        else:
        
            if "pilot_obs" in cfg.channel_prediction_method:
                rx_sig_freq_history, pilot_symbols = rc_predictor.get_pilot_history_with_metadata(
                    cfg.first_slot_idx,
                    cfg.csi_delay,
                    rg_csi,
                    dmimo_chans,
                    cfo_vals=cfg.random_cfo_vals,
                    sto_vals=cfg.random_sto_vals,
                    estimated_channels_dir=cfg.estimated_channels_dir,
                )
                err_var_csi_history = None
            elif "kalman" in cfg.channel_prediction_method or "configured_wesn" in cfg.channel_prediction_method:
                h_freq_csi_history, err_var_csi_history = rc_predictor.get_csi_history_with_err_var(
                    cfg.first_slot_idx,
                    cfg.csi_delay,
                    rg_csi,
                    dmimo_chans,
                    cfo_vals=cfg.random_cfo_vals,
                    sto_vals=cfg.random_sto_vals,
                    estimated_channels_dir=cfg.estimated_channels_dir,
                    freq_cov_mat=freq_cov_mat,
                    lmmse_interpolator=lmmse_interpolator,
                    use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar,
                )
                if getattr(cfg, "print_lmmse_effective_snr", True):
                    _print_lmmse_effective_snr(
                        "CSI history",
                        h_freq_csi_history,
                        err_var_csi_history,
                    )
            else:
                h_freq_csi_history = rc_predictor.get_csi_history(
                    cfg.first_slot_idx,
                    cfg.csi_delay,
                    rg_csi,
                    dmimo_chans,
                    cfo_vals=cfg.random_cfo_vals,
                    sto_vals=cfg.random_sto_vals,
                    estimated_channels_dir=cfg.estimated_channels_dir,
                    freq_cov_mat=freq_cov_mat,
                    lmmse_interpolator=lmmse_interpolator,
                    use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar,
                )
                err_var_csi_history = None
                
        end_time = time.time()
        # print("Total time for channel history gathering: ", end_time - start_time)

        h_freq_csi_perfect, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx,
                                                batch_size=cfg.num_slots_p2)
        # print("avg rx_snr_db: ", np.mean(rx_snr_db))
        
        start_time = time.time()
        if "pilot_obs" in cfg.channel_prediction_method:
            _, _, _, num_rx_ant, num_pilot_syms, num_freq = rx_sig_freq_history.shape
            # rx_sig_pred = rx_sig_predictor.predict(rx_sig_freq_history)
            rx_sig_pred = rx_sig_predict_all_links_simple(rx_sig_freq_history, rc_config, ns3cfg, err_var_csi_history=None)
            h_freq_csi, err_var_csi = estimate_channel_from_pilot_rx_symbols(
                rx_sig_pred,
                rg_csi,
                pilot_symbols,
                freq_cov_mat=freq_cov_mat,
                lmmse_interpolator=lmmse_interpolator,
            )
        elif "two_mode" in cfg.channel_prediction_method:

            if "two_mode_tf" in cfg.channel_prediction_method:

                start_time_all_loops = time.time()

                h_freq_csi = predict_all_links_tf(h_freq_csi_history, rc_config, ns3cfg, err_var_csi_history=err_var_csi_history)

                end_time_all_loops = time.time()
                # print("total time for prediction: ", end_time_all_loops-start_time_all_loops)

            else:
                start_time_all_loops = time.time()

                # h_freq_csi = predict_all_links(h_freq_csi_history, rc_config, ns3cfg, max_workers=8, err_var_csi_history=err_var_csi_history)
                h_freq_csi = predict_all_links_simple(h_freq_csi_history, rc_config, ns3cfg, err_var_csi_history=err_var_csi_history)

                end_time_all_loops = time.time()
                # print("total time for prediction: ", end_time_all_loops-start_time_all_loops)
        elif "configured_wesn" in cfg.channel_prediction_method:
            configured_predictors = getattr(cfg, "configured_wesn_predictors", None)
            if configured_predictors is None:
                raise ValueError(
                    "Configured WESN predictors not found. "
                    "Please configure offline predictors before running online slots."
                )
            h_freq_csi = predict_all_links_with_configured_simple(
                h_freq_csi_history,
                configured_predictors,
                ns3cfg,
                err_var_csi_history=err_var_csi_history,
            )
        elif "channelmamba" in cfg.channel_prediction_method:
            from dmimo.channel.channelmamba_pred import predict_all_links_with_channelmamba_per_pair

            channelmamba_predictors = getattr(cfg, "channelmamba_predictors", None)
            if channelmamba_predictors is None:
                raise ValueError(
                    "ChannelMamba predictors not found. "
                    "Please run offline ChannelMamba training before online slots."
                )
            h_freq_csi = predict_all_links_with_channelmamba_per_pair(
                h_freq_csi_history,
                channelmamba_predictors,
                ns3cfg,
            )

        elif cfg.channel_prediction_method == "old":
            h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
        elif "weiner_filter" in cfg.channel_prediction_method:
            # Weiner Filter based prediction (MIMO) (per_tx_rx_node_pair)
            weiner_filter_predictor = weiner_filter_pred(method="using_one_link_MIMO")
            h_freq_csi = np.asarray(weiner_filter_predictor.predict(h_freq_csi_history, K=rc_config.window_length))
        elif "kalman" in cfg.channel_prediction_method:
            if err_var_csi_history is None:
                raise ValueError("Kalman predictor requires measurement error variance history.")

            h_freq_csi_history_perfect = rc_predictor.get_ideal_csi_history(cfg.first_slot_idx+cfg.csi_delay, cfg.csi_delay,
                                                          dmimo_chans)
            kalman_predictor = kalman_filter_pred(ar_order=rc_config.window_length, debug=False)
            h_freq_csi = kalman_predictor.predict(
                h_freq_csi_history,
                err_var_csi_history,
                h_freq_csi_perfect_debug=h_freq_csi_history_perfect,
            )
        else:
            raise ValueError("Channel prediction method not implemented here.")
        end_time = time.time()
        # print("{} Prediction Time: {}".format(cfg.channel_prediction_method, end_time - start_time))
    else:
        h_freq_csi_perfect, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx,
                                                batch_size=cfg.num_slots_p2)
        
        # LMMSE channel estimation. h_freq_csi shape: [_, _, num_rx_ants, _ num_tx_ants, num_syms, num_subcarriers]
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals,
                                                           freq_cov_mat=freq_cov_mat,
                                                           lmmse_interpolator=lmmse_interpolator,
                                                           use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar)
        if getattr(cfg, "print_lmmse_effective_snr", True):
            _print_lmmse_effective_snr("Current slot", h_freq_csi, err_var_csi)
    
    chan_pred_nmse = tf.reduce_mean(tf.abs(h_freq_csi_perfect[0:1,...] - h_freq_csi[0:1,...])**2) / tf.reduce_mean(tf.abs(h_freq_csi_perfect[0:1,...])**2)
    print("{} Prediction NMSE: {}".format(cfg.channel_prediction_method, chan_pred_nmse))
    
    # Pick the selected UE's channels
    h_freq_csi = tf.gather(h_freq_csi, tf.reshape(cfg.scheduled_rx_ue_indices, (-1,)), axis=2)
    h_freq_csi = tf.gather(h_freq_csi, tf.reshape(cfg.scheduled_tx_ue_indices, (-1,)), axis=4)
    h_freq_csi_unquantized = h_freq_csi
        
    PMI_feedback_bits = None
    mcs_indices = None

    if cfg.csi_quantization_on:
        h_freq_csi_unquantized = tf.reduce_mean(h_freq_csi_unquantized, axis=0, keepdims=True)
        if cfg.PMI_feedback_architecture == "RVQ":
            rvq = RandomVectorQuantizer(bits_per_codeword=15, vector_dim=h_freq_csi.shape[4], seed=42)
            h_freq_csi = rvq.quantize_feedback(h_freq_csi, cfg, rg_csi, donald_hack=True, quantization_debug=False)
        else:
            type_II_PMI_quantizer = quantized_CSI_feedback(method='5G', 
                                                            codebook_selection_method=None,
                                                            num_tx_streams=cfg.num_tx_streams,
                                                            architecture=cfg.PMI_feedback_architecture,
                                                            rbs_per_subband=4,
                                                            snrdb=rx_snr_db)
            h_freq_csi, PMI_feedback_bits = type_II_PMI_quantizer(
                h_freq_csi_unquantized,
                return_feedback_bits=True,
            )

            w1_override = None
            if cfg.csi_prediction is True and "deqn" in cfg.channel_prediction_method:
                rl_selector = getattr(cfg, "rl_selector", None)
                last_mcs_indices = getattr(cfg, "last_mcs_indices", None)
                last_node_wise_acks = getattr(cfg, "last_node_wise_acks", None)
                last_throughput = getattr(cfg, "last_throughput", None)
                last_target_throughput = getattr(cfg, "last_target_throughput", None)
                if rl_selector is not None and last_mcs_indices is not None and last_node_wise_acks is not None:
                    w1_override = rl_selector.prepare_next_actions(
                        PMI_feedback_bits,
                        mcs_indices=last_mcs_indices,
                        node_wise_acks=last_node_wise_acks,
                        throughput_debug=last_throughput,
                        num_transitions=cfg.num_transitions,
                        no_rl_throughput=cfg.curr_no_rl_throughput
                    )

            if w1_override is not None:
                h_freq_csi, _ = type_II_PMI_quantizer(
                    h_freq_csi_unquantized,
                    return_feedback_bits=True,
                    w1_beam_indices_override=w1_override,
                )
                
            h_freq_csi = tf.squeeze(h_freq_csi, axis=(1,3))

    ranks_out = int(cfg.num_tx_streams / (cfg.num_scheduled_ues+2))

    # Create MU-MIMO simulation
    mu_mimo = MU_MIMO(cfg, rg_csi)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    info_bits = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    # Saving Rx SNRs
    rx_snr_lin = 10.0 **( rx_snr_db / 10.0)
    rx_snr_lin = np.mean(rx_snr_lin, axis=(0,1, 3))
    rx_snr_lin = np.reshape(rx_snr_lin, [ns3cfg.num_rxue_sel+2, -1])
    rx_snr_lin = np.mean(rx_snr_lin, axis=-1)
    snr_dB_arr = 10*np.log10(rx_snr_lin)

    # MU-MIMO transmission (P2)
    dec_bits, uncoded_ber_phase_2, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_db_arr = mu_mimo(dmimo_chans, h_freq_csi, info_bits, snr_dB_arr)
    
    if cfg.rank_adapt or cfg.link_adapt:
        # Rank and link adaptation
        # TODO: add support for quantized CSI feedback
        rank_feedback_report, mcs_feedback_report = \
            do_rank_link_adaptation(cfg, dmimo_chans, h_freq_csi_unquantized, sinr_db_arr, return_mcs_index=True)

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
        cqi_sinrs = mcs_feedback_report[2]
        mcs_indices = mcs_feedback_report[3]
        values, counts = np.unique(qam_order_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.modulation_order = int(most_frequent_value)

        values, counts = np.unique(code_rate_arr, return_counts=True)
        most_frequent_value = values[np.argmax(counts)]
        cfg.code_rate = most_frequent_value

        # print("\n", "Bits per stream per user (MU-MIMO) = ", cfg.modulation_order)
        # print("Code-rate per stream per user (MU-MIMO) = ", cfg.code_rate, "\n")

    # Update error statistics
    if bool(getattr(cfg, "return_first_slot_only", False)):
        info_bits = tf.reshape(info_bits[0:1, ...], dec_bits.shape) # shape: [batch_size, 1, num_streams_per_tx, num_codewords, num_effective_subcarriers*num_data_ofdm_syms_per_subframe]
    else:
        info_bits = tf.reshape(info_bits, dec_bits.shape) # shape: [batch_size, 1, num_streams_per_tx, num_codewords, num_effective_subcarriers*num_data_ofdm_syms_per_subframe]
    coded_ber = compute_ber(info_bits, dec_bits).numpy()
    coded_bler = compute_bler(info_bits, dec_bits).numpy()
    print("Uncoded BER: ", uncoded_ber_phase_2)
    # print("Coded BER: ", coded_ber)
    print("BLER: ", coded_bler)

    node_wise_ber, node_wise_bler = compute_UE_wise_BER(info_bits, dec_bits, cfg.ue_ranks[0], cfg.num_tx_streams)
    node_wise_acks = 1 - np.ceil(node_wise_bler)
    cfg.last_mcs_indices = mcs_indices
    cfg.last_node_wise_acks = node_wise_acks

    # Goodput and throughput estimation
    goodbits = (1.0 - coded_ber) * mu_mimo.num_bits_per_frame
    userbits = (1.0 - coded_bler) * mu_mimo.num_bits_per_frame
    ratedbits_phase_2 = (1.0 - uncoded_ser) * mu_mimo.num_uncoded_bits_per_frame

    overhead = cfg.num_slots_p2/(cfg.num_slots_p1 + cfg.num_slots_p2)
    cfg.last_throughput = userbits / (cfg.slot_duration * 1e6) * overhead  # Mbps
    cfg.last_target_throughput = mu_mimo.num_bits_per_frame / (cfg.slot_duration * 1e6) * overhead  # Mbps
    # print("cfg.last_throughput: ", cfg.last_throughput)
    # print("cfg.last_target_throughput: ", cfg.last_target_throughput)

    node_wise_goodbits_phase_2 = (1.0 - node_wise_ber) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_userbits_phase_2 = (1.0 - node_wise_bler) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)
    node_wise_ratedbits_phase_2 = (1.0 - node_wise_uncoded_ser) * mu_mimo.num_bits_per_frame / (cfg.num_scheduled_ues + 1)

    return [uncoded_ber_phase_2, coded_ber], [goodbits, userbits, ratedbits_phase_2], [node_wise_goodbits_phase_2, node_wise_userbits_phase_2, node_wise_ratedbits_phase_2, ranks_out, sinr_db_arr, snr_dB_arr, PMI_feedback_bits, node_wise_bler, chan_pred_nmse]


def sim_mu_mimo_all(
    cfg: SimConfig,
    ns3cfg: Ns3Config,
    rc_config:RCConfig,
    rl_selector: Optional[RLBeamSelector] = None,
):
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
    snr_dB_list = []
    PMI_feedback_bits = []
    nodewise_bler_list = []
    per_step_throughput = []
    chan_pred_nmse = []

    cfg.curr_no_rl_throughput = None
    cfg.rl_selector = rl_selector

    slot_indices_all = np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2)
    slot_indices = slot_indices_all.copy()

    eval_on_online_segment_only = bool(getattr(cfg, "eval_on_online_segment_only", True))
    cfg.eval_on_online_segment_only = eval_on_online_segment_only

    is_configured_wesn = cfg.csi_prediction and "configured_wesn" in str(cfg.channel_prediction_method)
    is_kalman_filter = cfg.csi_prediction and "kalman_filter" in str(cfg.channel_prediction_method)
    is_channelmamba = cfg.csi_prediction and "channelmamba" in str(cfg.channel_prediction_method)

    offline_ratio = float(getattr(cfg, "wesn_offline_ratio", 0.5))
    if not (0.0 < offline_ratio <= 1.0):
        raise ValueError(f"offline ratio must be in (0, 1], got {offline_ratio}.")

    if is_channelmamba:
        offline_cycles = 0
    elif slot_indices_all.size <= 1:
        offline_cycles = 0
    else:
        offline_cycles = int(np.floor(slot_indices_all.size * offline_ratio))
        offline_cycles = max(1, min(offline_cycles, slot_indices_all.size - 1))
    offline_slot_indices = slot_indices_all[:offline_cycles]
    num_offline_cycles = int(offline_cycles)
    num_online_cycles = int(slot_indices_all.size - offline_cycles)

    if eval_on_online_segment_only and is_kalman_filter:
        if slot_indices_all.size <= 1:
            raise ValueError("Kalman filter evaluation requires at least two cycles (offline + online).")
        slot_indices = slot_indices_all[offline_cycles:]

    if is_configured_wesn:
        configured_wesn_split_mode = str(getattr(cfg, "configured_wesn_split_mode", "within_drop")).lower()
        if configured_wesn_split_mode not in ("within_drop", "across_drops"):
            raise ValueError(
                f"Unsupported configured_wesn_split_mode='{configured_wesn_split_mode}'. "
                "Expected 'within_drop' or 'across_drops'."
            )
        if slot_indices_all.size <= 1:
            raise ValueError("Configured WESN requires at least two cycles (offline + online).")
        if eval_on_online_segment_only and configured_wesn_split_mode == "within_drop":
            slot_indices = slot_indices_all[offline_cycles:]

        # Reset UE selection. Start with all TX and RX UEs selected.
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
        else:
            raise Exception ("Scheduling not supported in this version.")

        dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
        num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant
        csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1
        rg_csi = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=num_txs_ant,
            cyclic_prefix_length=cfg.cyclic_prefix_len,
            num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        rc_predictor = getattr(cfg, "rc_predictor", None)
        if rc_predictor is None:
            rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams, rc_config, ns3cfg)
            cfg.rc_predictor = rc_predictor
        rc_predictor.reset_csi_history()

        freq_cov_mat = getattr(cfg, "freq_cov_mat", None)
        lmmse_interpolator = getattr(cfg, "lmmse_interpolator", None)
        lmmse_use_rx_snr_for_nvar = getattr(cfg, "lmmse_use_rx_snr_for_nvar", True)
        if configured_wesn_split_mode == "within_drop":
            first_slot_idx = slot_indices_all[offline_cycles]
            rc_predictor.history_len = offline_slot_indices.size
            h_hist, err_hist = rc_predictor.get_csi_history_with_err_var(
                first_slot_idx,
                cfg.csi_delay,
                rg_csi,
                dmimo_chans,
                cfo_vals=cfg.random_cfo_vals,
                sto_vals=cfg.random_sto_vals,
                estimated_channels_dir=cfg.estimated_channels_dir,
                freq_cov_mat=freq_cov_mat,
                lmmse_interpolator=lmmse_interpolator,
                use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar,
            )
            offline_history = np.asarray(h_hist)
            offline_err_history = np.asarray(err_hist)
        else:
            train_drop_indices = getattr(cfg, "configured_wesn_train_drop_indices", None)
            if train_drop_indices is None:
                train_drop_indices = [str(cfg.drop_idx)]
            else:
                train_drop_indices = [str(d) for d in train_drop_indices]

            print(f"[configured_wesn] drop={cfg.drop_idx}: pooled offline train drops={train_drop_indices}")
            pooled_h_hist_per_drop = []
            pooled_err_hist_per_drop = []
            original_ns3_folder = cfg.ns3_folder
            original_estimated_channels_dir = cfg.estimated_channels_dir
            old_history_len = int(rc_predictor.history_len)
            try:
                for train_drop_idx in train_drop_indices:
                    cfg.ns3_folder = f"ns3/channels_{cfg.mobility}_{train_drop_idx}/"
                    cfg.estimated_channels_dir = f"ns3/channel_estimates_{cfg.mobility}_drop_{train_drop_idx}"
                    dmimo_chans_train = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
                    per_drop_h_hist = []
                    per_drop_err_hist = []
                    rc_predictor.history_len = 1
                    rc_predictor.reset_csi_history()
                    for hist_slot_idx in slot_indices_all:
                        h_hist_one, err_hist_one = rc_predictor.get_csi_history_with_err_var(
                            int(hist_slot_idx),
                            cfg.csi_delay,
                            rg_csi,
                            dmimo_chans_train,
                            cfo_vals=cfg.random_cfo_vals,
                            sto_vals=cfg.random_sto_vals,
                            estimated_channels_dir=cfg.estimated_channels_dir,
                            freq_cov_mat=freq_cov_mat,
                            lmmse_interpolator=lmmse_interpolator,
                            use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar,
                        )
                        per_drop_h_hist.append(np.asarray(h_hist_one))
                        per_drop_err_hist.append(np.asarray(err_hist_one))
                    if per_drop_h_hist:
                        pooled_h_hist_per_drop.append(np.concatenate(per_drop_h_hist, axis=0))
                        pooled_err_hist_per_drop.append(np.concatenate(per_drop_err_hist, axis=0))
            finally:
                cfg.ns3_folder = original_ns3_folder
                cfg.estimated_channels_dir = original_estimated_channels_dir
                rc_predictor.history_len = old_history_len
                rc_predictor.reset_csi_history()

            if len(pooled_h_hist_per_drop) == 0 or len(pooled_err_hist_per_drop) == 0:
                raise ValueError(
                    f"Configured WESN pooled training produced no CSI history blocks for drops={train_drop_indices}."
                )
            offline_history = np.concatenate(pooled_h_hist_per_drop, axis=0)
            offline_err_history = np.concatenate(pooled_err_hist_per_drop, axis=0)
            slot_indices = slot_indices_all
        cfg.configured_wesn_predictors = build_configured_predictors_simple(
            offline_history,
            rc_config,
            ns3cfg,
            err_var_csi_history=offline_err_history,
        )

    if is_channelmamba:
        from dmimo.channel.channelmamba_pred import build_channelmamba_predictors_simple

        if slot_indices_all.size <= 0:
            raise ValueError("ChannelMamba evaluation requires at least one cycle.")
        
        # Reset UE selection. Start with all TX and RX UEs selected.
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
        else:
            raise Exception ("Scheduling not supported in this version.")

        dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
        num_txs_ant = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant
        csi_effective_subcarriers = (cfg.fft_size // num_txs_ant) * num_txs_ant
        csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
        csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1
        rg_csi = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=num_txs_ant,
            cyclic_prefix_length=cfg.cyclic_prefix_len,
            num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
            dc_null=False,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=[2, 11],
        )

        rc_predictor = getattr(cfg, "rc_predictor", None)
        if rc_predictor is None:
            rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams, rc_config, ns3cfg)
            cfg.rc_predictor = rc_predictor
        rc_predictor.reset_csi_history()

        freq_cov_mat = getattr(cfg, "freq_cov_mat", None)
        lmmse_interpolator = getattr(cfg, "lmmse_interpolator", None)
        lmmse_use_rx_snr_for_nvar = getattr(cfg, "lmmse_use_rx_snr_for_nvar", True)

        channelmamba_mode = str(getattr(cfg, "channelmamba_mode", "train")).lower()
        channelmamba_prev_len = int(getattr(cfg, "channelmamba_prev_len", 16))
        channelmamba_pred_len = int(getattr(cfg, "channelmamba_pred_len", 1))
        if channelmamba_prev_len <= 0 or channelmamba_pred_len <= 0:
            raise ValueError(
                f"ChannelMamba requires positive prev_len/pred_len, got prev_len={channelmamba_prev_len}, "
                f"pred_len={channelmamba_pred_len}."
            )

        # Build an offline history timeline aligned with simulation cycles:
        # input slots are the previous `channelmamba_prev_len` delayed CSI estimates and
        # target is the next delayed-step CSI estimate.
        # Example (csi_delay=4, prev_len=5, pred_len=1):
        #   [13,17,21,25,29] -> 33, [17,21,25,29,33] -> 37, ...
        first_label_slot_idx = int(slot_indices_all[0])
        last_label_slot_idx = int(slot_indices_all[-1])
        first_history_slot_idx = first_label_slot_idx - cfg.csi_delay * channelmamba_prev_len
        last_required_slot_idx = last_label_slot_idx + cfg.csi_delay * (channelmamba_pred_len - 1)
        channelmamba_slots = np.arange(
            first_history_slot_idx,
            last_required_slot_idx + cfg.csi_delay,
            cfg.csi_delay,
            dtype=int,
        )
        if channelmamba_slots.size < (channelmamba_prev_len + channelmamba_pred_len):
            raise ValueError(
                "Insufficient ChannelMamba offline timeline slots. "
                f"Need at least {channelmamba_prev_len + channelmamba_pred_len}, "
                f"got {channelmamba_slots.size}."
            )
        train_drop_indices = getattr(cfg, "channelmamba_train_drop_indices", None)
        if train_drop_indices is None:
            train_drop_indices = [str(cfg.drop_idx)]
        else:
            train_drop_indices = [str(d) for d in train_drop_indices]

        channelmamba_split_mode = str(getattr(cfg, "channelmamba_split_mode", "drop_split")).lower()
        channelmamba_train_time_ratio = float(getattr(cfg, "channelmamba_train_time_ratio", 1.0))
        if not (0.0 < channelmamba_train_time_ratio <= 1.0):
            raise ValueError(
                f"channelmamba_train_time_ratio must be in (0, 1], got {channelmamba_train_time_ratio}."
            )
        if channelmamba_split_mode == "time_split":
            num_timeline = int(channelmamba_slots.size)
            train_timeline_count = int(np.floor(num_timeline * channelmamba_train_time_ratio))
            train_timeline_count = max(1, min(train_timeline_count, num_timeline))
            train_channelmamba_slots = channelmamba_slots[:train_timeline_count]
            print(
                f"[channelmamba] drop={cfg.drop_idx}: time-split training timeline "
                f"count={train_timeline_count}/{num_timeline} (ratio={channelmamba_train_time_ratio})."
            )
        else:
            train_channelmamba_slots = channelmamba_slots

        if channelmamba_mode in ("train", "auto"):
            if channelmamba_mode == "train":
                print(f"[channelmamba] drop={cfg.drop_idx}: training and optionally saving to {cfg.channelmamba_checkpoint}")
            else:
                print(f"[channelmamba] drop={cfg.drop_idx}: auto mode with checkpoint={cfg.channelmamba_checkpoint}")
            print(f"[channelmamba] drop={cfg.drop_idx}: pooled offline train drops={train_drop_indices}")

            pooled_h_hist_per_drop = []
            original_ns3_folder = cfg.ns3_folder
            original_estimated_channels_dir = cfg.estimated_channels_dir

            old_history_len = int(rc_predictor.history_len)
            try:
                for train_drop_idx in train_drop_indices:
                    cfg.ns3_folder = f"ns3/channels_{cfg.mobility}_{train_drop_idx}/"
                    cfg.estimated_channels_dir = f"ns3/channel_estimates_{cfg.mobility}_drop_{train_drop_idx}"
                    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
                    rc_predictor.history_len = 1
                    rc_predictor.reset_csi_history()
                    per_drop_blocks = []
                    for hist_slot_idx in train_channelmamba_slots:
                        # With history_len=1, querying at first_slot_idx=(hist_slot_idx + csi_delay)
                        # returns exactly the CSI estimate at hist_slot_idx.
                        h_hist_one, _ = rc_predictor.get_csi_history_with_err_var(
                            int(hist_slot_idx + cfg.csi_delay),
                            cfg.csi_delay,
                            rg_csi,
                            dmimo_chans,
                            cfo_vals=cfg.random_cfo_vals,
                            sto_vals=cfg.random_sto_vals,
                            estimated_channels_dir=cfg.estimated_channels_dir,
                            freq_cov_mat=freq_cov_mat,
                            lmmse_interpolator=lmmse_interpolator,
                            use_rx_snr_for_nvar=lmmse_use_rx_snr_for_nvar,
                        )
                        per_drop_blocks.append(np.asarray(h_hist_one))
                    if len(per_drop_blocks) == 0:
                        continue
                    pooled_h_hist_per_drop.append(np.concatenate(per_drop_blocks, axis=0))
            finally:
                cfg.ns3_folder = original_ns3_folder
                cfg.estimated_channels_dir = original_estimated_channels_dir
                rc_predictor.history_len = old_history_len
                rc_predictor.reset_csi_history()

            if len(pooled_h_hist_per_drop) == 0:
                raise ValueError(
                    f"ChannelMamba pooled training produced no CSI history blocks for drops={train_drop_indices}."
                )
            channelmamba_predictors = build_channelmamba_predictors_simple(
                pooled_h_hist_per_drop,
                cfg,
                ns3cfg,
            )
        elif channelmamba_mode == "eval":
            if not cfg.channelmamba_checkpoint:
                raise ValueError("channelmamba_mode='eval' requires cfg.channelmamba_checkpoint to be set.")
            print(f"[channelmamba] drop={cfg.drop_idx}: loading pair checkpoints from base {cfg.channelmamba_checkpoint}")
            channelmamba_predictors = build_channelmamba_predictors_simple(
                [],
                cfg,
                ns3cfg,
            )
            print(f"[channelmamba] drop={cfg.drop_idx}: skipped offline history extraction in eval mode")
        else:
            raise ValueError(f"Unsupported channelmamba_mode='{channelmamba_mode}'.")
        cfg.channelmamba_predictors = channelmamba_predictors

    if eval_on_online_segment_only and is_kalman_filter:
        online_loop_slot_indices = slot_indices_all[offline_cycles + 1:]
    else:
        online_loop_slot_indices = slot_indices[1:]
    print(
        f"Drop {cfg.drop_idx}, {cfg.channel_prediction_method} online loop slot indices: "
        f"{online_loop_slot_indices.tolist()}"
    )

    cfg.start_slot_idx = slot_indices[0]
    
    for first_slot_idx in slot_indices:
        
        # print("first_slot_idx: ", first_slot_idx)

        cfg.first_slot_idx = first_slot_idx

        start_time = time.time()
        bers, bits, additional_KPIs = sim_mu_mimo(cfg, ns3cfg, rc_config)
        end_time = time.time()
        # print("Cycle time: ", end_time - start_time, " seconds\n")
        
        use_cycle_for_avg = first_slot_idx > cfg.start_slot_idx

        if use_cycle_for_avg:
            
            total_cycles += 1
            
            uncoded_ber += bers[0]
            ldpc_ber += bers[1]
            uncoded_ber_list.append(bers[0])
            ldpc_ber_list.append(bers[1])
            
            goodput += bits[0]
            throughput += bits[1]
            bitrate += bits[2]
            per_step_throughput.append(bits[1] / (slot_time * 1e6) * overhead)
            
            nodewise_goodput.append(additional_KPIs[0])
            nodewise_throughput.append(additional_KPIs[1])
            nodewise_bitrate.append(additional_KPIs[2])
            ranks_list.append(additional_KPIs[3])
            sinr_dB_list.append(additional_KPIs[4])
            snr_dB_list.append(additional_KPIs[5])
            PMI_feedback_bits.append(additional_KPIs[6])
            nodewise_bler_list.append(additional_KPIs[7])
            chan_pred_nmse.append(additional_KPIs[8])

        hold = 1

    num_cycles_used_for_avg = int(total_cycles)
    print(
        f"Drop {cfg.drop_idx}, {cfg.channel_prediction_method} cycle counts: "
        f"num_offline_cycles={num_offline_cycles}, "
        f"num_online_cycles={num_online_cycles}, "
        f"num_cycles_used_for_avg={num_cycles_used_for_avg}"
    )
    if num_cycles_used_for_avg <= 0:
        raise ValueError("No cycles available for averaging after applying evaluation filters.")

    goodput = goodput / (num_cycles_used_for_avg * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (num_cycles_used_for_avg * slot_time * 1e6) * overhead  # Mbps
    bitrate = bitrate / (num_cycles_used_for_avg * slot_time * 1e6) * overhead  # Mbps

    # print("Average throughput: {:.2f} Mbps".format(throughput))
    # print("Average uncoded BER: {:.2f}".format(uncoded_ber / total_cycles))
    # print("Average coded BER: {:.2f}".format(ldpc_ber / total_cycles))

    nodewise_goodput = np.concatenate(nodewise_goodput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_throughput = np.concatenate(nodewise_throughput) / (slot_time * 1e6) * overhead  # Mbps
    nodewise_bitrate = np.concatenate(nodewise_bitrate) / (slot_time * 1e6) * overhead  # Mbps
    ranks = np.array(ranks_list).flatten()
    if sinr_dB_list[0] is not None:
        sinr_dB = np.concatenate(sinr_dB_list)
    else:
        sinr_dB = None

    if snr_dB_list[0] is not None:
        snr_dB = np.concatenate(snr_dB_list)
    else:
        snr_dB = None

    if rl_selector is not None:
        checkpoint_dir = Path("results") / "deqn_checkpoints" / Path(cfg.ns3_folder.rstrip("/")).name
        rl_selector.save_all(checkpoint_dir)

    per_step_throughput = np.array(per_step_throughput)
    chan_pred_nmse = np.array(chan_pred_nmse)

    print("Drop {}, {} Average Prediction NMSE: {}".format(cfg.drop_idx, cfg.channel_prediction_method, np.mean(chan_pred_nmse)))
    print("Drop {}, Average throughput: {:.2f} Mbps".format(cfg.drop_idx, throughput))
    print("Drop {}, Average uncoded BER: {:.2f}\n".format(cfg.drop_idx, uncoded_ber / num_cycles_used_for_avg))

    return [
        uncoded_ber / num_cycles_used_for_avg,
        ldpc_ber / num_cycles_used_for_avg,
        goodput,
        throughput,
        bitrate,
        nodewise_goodput,
        nodewise_throughput,
        nodewise_bitrate,
        ranks,
        uncoded_ber_list,
        ldpc_ber_list,
        sinr_dB,
        snr_dB,
        per_step_throughput,
        chan_pred_nmse
    ]