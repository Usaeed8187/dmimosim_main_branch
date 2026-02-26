import copy
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
from dmimo.channel import dMIMOChannels, lmmse_channel_estimation, estimate_freq_cov, LMMSELinearInterp
from dmimo.channel import standard_rc_pred_freq_mimo#, default_ddpg_predictor
# from dmimo.channel.ddpg_predictor import DDPGChannelPredictor
from dmimo.channel import twomode_wesn_pred, twomode_wesn_pred_tf, weiner_filter_pred
from dmimo.channel.twomode_wesn_pred import predict_all_links, predict_all_links_simple
from dmimo.channel.rl_beam_selector_v2 import RLBeamSelector
from dmimo.channel.twomode_wesn_pred_tf import predict_all_links_tf
from dmimo.channel import RBwiseLinearInterp
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, QuantizedSLNRPrecoder, SLNREqualizer, QuantizedZFPrecoder, QuantizedDirectPrecoder
from dmimo.mimo import rankAdaptation, linkAdaptation
from dmimo.mimo import MUMIMOScheduler
from dmimo.mimo import update_node_selection, quantized_CSI_feedback, RandomVectorQuantizer, RandomVectorQuantizerNumpy
from dmimo.phase_1 import Phase1v
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER, complex_pinv

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

    def call(self, dmimo_chans: dMIMOChannels, h_freq_csi, info_bits_list, snr_dB_arr):
        """
        Signal processing for one MU-MIMO transmission cycle (P2)

        :param dmimo_chans: dMIMO channels
        :param h_freq_csi: CSI feedback for precoding
        :param info_bits_list: list of information bits for each transmitter
        :return: decoded bits, uncoded BER, demodulated QAM symbols (for debugging purpose)
        """

        x_precoded_list = []
        for i in range(len(info_bits_list)):
            info_bits = info_bits_list[i]
            start_ant_idx = 0 if i == 0 else 4 + 2*(i-1) # assuming 4 BS antennas and 2 antennas per UE, adjust if different
            end_ant_idx = 4 if i == 0 else start_ant_idx + 2

            # LDPC encoder processing
            info_bits = tf.reshape(info_bits, [self.batch_size, 1, self.rg.num_streams_per_tx,
                                            self.num_codewords, self.encoder.k])
            c = self.encoder(info_bits)
            c = tf.reshape(c, [self.batch_size, 1, self.rg.num_streams_per_tx, self.num_codewords * self.encoder.n])

            # Interleaving for coded bits
            d_this_transmitter = self.intlvr(c)
            if i==0: # If this transmitter is the base station
                d = d_this_transmitter # because the first transmitter's bits are always error free

            # QAM mapping for the OFDM grid
            x = self.mapper(d_this_transmitter)
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

            # x_precoded has shape [self.batch_size, num_tx, num_tx_ant, num_syms, num_sc] 
            # e.g. = [3, 1, 20, 14, 512] 
            x_precoded_list.append(x_precoded[:,:, start_ant_idx:end_ant_idx, :, :])

        # concatenate precoded signals for different transmitters, shape [self.batch_size, num_tx, num_txs_ant, num_syms, num_sc]
        x_precoded = tf.concat(x_precoded_list, axis=2) 
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


    # Total number of antennas in the TxSquad, always use all gNB antennas
    num_txs_ant_p2 = 2 * ns3cfg.num_txue_sel + ns3cfg.num_bs_ant # -> Ramin edit: need to distinguish between phase 1 and phase 2 number of Tx antennas.
    

    # dMIMO channels from ns-3 simulator
    dmimo_chans = dMIMOChannels(ns3cfg, "dMIMO", add_noise=True, return_channel=True)
    
    

    # Adjust guard subcarriers for channel estimation grid
    csi_effective_subcarriers = (cfg.fft_size // num_txs_ant_p2) * num_txs_ant_p2 # -> Ramind edit: replace num_txs_ant with num_txs_ant_p2.
    csi_guard_carriers_1 = (cfg.fft_size - csi_effective_subcarriers) // 2
    csi_guard_carriers_2 = (cfg.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

    # Resource grid for channel estimation
    rg_csi = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg.fft_size,
                          subcarrier_spacing=cfg.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_txs_ant_p2, # -> Ramin edit
                          cyclic_prefix_length=cfg.cyclic_prefix_len,
                          num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                          dc_null=False,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])
    
    # Cacheable LMMSE resources for the current drop
    freq_cov_mat = getattr(cfg, "freq_cov_mat", None)
    lmmse_interpolator = getattr(cfg, "lmmse_interpolator", None)

    # Channel CSI estimation using channels in previous frames/slots
    h_freq_csi_history = None
    if cfg.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi, rx_snr_db, rx_pwr_dbm = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx,
                                                                     batch_size=cfg.num_slots_p2)
    elif cfg.csi_prediction is True:
        rc_predictor = getattr(cfg, "rc_predictor", None)

        if rc_predictor is None:
            rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams, rc_config, ns3cfg)
            cfg.rc_predictor = rc_predictor
        if cfg.first_slot_idx == cfg.start_slot_idx:
            rc_predictor.reset_csi_history()
        # Get CSI history
        # TODO: optimize channel estimation and optimization procedures (currently very slow)        

        start_time = time.time()
        if cfg.use_perfect_csi_history_for_prediction:
            h_freq_csi_history = rc_predictor.get_ideal_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                          dmimo_chans)
        else:
        
            h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
                                                            rg_csi, dmimo_chans, 
                                                            cfo_vals=cfg.random_cfo_vals,
                                                            sto_vals=cfg.random_sto_vals,
                                                            estimated_channels_dir=cfg.estimated_channels_dir,
                                                            freq_cov_mat=freq_cov_mat,
                                                            lmmse_interpolator=lmmse_interpolator)
                
        end_time = time.time()
        # print("Total time for channel history gathering: ", end_time - start_time)
        
        if "two_mode" in cfg.channel_prediction_method:

            if "two_mode_tf" in cfg.channel_prediction_method:

                start_time_all_loops = time.time()

                h_freq_csi = predict_all_links_tf(h_freq_csi_history, rc_config, ns3cfg)

                end_time_all_loops = time.time()
                # print("total time for prediction: ", end_time_all_loops-start_time_all_loops)

            else:
                start_time_all_loops = time.time()

                h_freq_csi = predict_all_links(h_freq_csi_history, rc_config, ns3cfg, max_workers=8)
                # h_freq_csi = predict_all_links_simple(h_freq_csi_history, rc_config, ns3cfg)

                end_time_all_loops = time.time()
                # print("total time for prediction: ", end_time_all_loops-start_time_all_loops)

        elif cfg.channel_prediction_method == "old":
            h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
        elif "weiner_filter" in cfg.channel_prediction_method:
            # Weiner Filter based prediction (MIMO) (per_tx_rx_node_pair)
            weiner_filter_predictor = weiner_filter_pred(method="using_one_link_MIMO")
            h_freq_csi = np.asarray(weiner_filter_predictor.predict(h_freq_csi_history, K=rc_config.history_len-1))
        elif cfg.channel_prediction_method == "ddpg":
            ddpg_actions = getattr(cfg, "ddpg_pred_channel", None)
            if ddpg_actions is None:
                h_freq_csi = h_freq_csi_history[-1, ...]
            else:
                h_freq_csi = ddpg_actions
        else:
            raise ValueError("Channel prediction method not implemented here.")
    else:
        # LMMSE channel estimation. h_freq_csi shape: [_, _, num_rx_ants, _ num_tx_ants, num_syms, num_subcarriers]
        h_freq_csi, err_var_csi = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                           slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                           cfo_vals=cfg.random_cfo_vals,
                                                           sto_vals=cfg.random_sto_vals,
                                                           freq_cov_mat=freq_cov_mat,
                                                           lmmse_interpolator=lmmse_interpolator)
    
    _, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=cfg.first_slot_idx - cfg.csi_delay,
                                                batch_size=cfg.num_slots_p2)
    
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
    ### Start Ramin edit: Phase 1 parameters and quantized channel feedback
    
    # dMIMO channels from ns-3 simulator
    p1_chans_dl = dMIMOChannels(ns3cfg, "TxSquad", forward=True, add_noise=True)

    # Total number of TX (gNB) antennas in the TxSquad
    num_txs_ant_p1 = ns3cfg.num_bs_ant  # always use all gNB antennas for precoding in phase 1
    # Least common multiple of number of TX antennas in phase 1 and phase 2, used for adjusting the resource grid
    # lcm_txs_ant = np.lcm(num_txs_ant_p1, num_txs_ant_p2) # This ensures the number of info bits is the same in phase 1 and phase 2
    cfg_p1 = copy.deepcopy(cfg)
    # The Tx gNB can broadcast 1 or 2 streams in phase 1.
    
    ## Hard coded parameters. TODO: make them configurable later. 
    # We choose 2 streams for phase 1. 
    cfg_p1.num_tx_streams = 2   
    cfg_p1.dc_null = False 
    # Number of scheduled UEs in phase 1 is cfg.scheduled_tx_ue_indices - 2 because we assume the first 2 UEs in scheduled_tx_ue_indices are the gNB's own antennas
    cfg_p1.num_scheduled_tx_ue = len(cfg.scheduled_tx_ue_indices) - 2

    # Adjust guard subcarriers for channel estimation grid
    csi_effective_subcarriers = (cfg_p1.fft_size // num_txs_ant_p1) * num_txs_ant_p1
    csi_guard_carriers_1 = (cfg_p1.fft_size - csi_effective_subcarriers) // 2
    csi_guard_carriers_2 = (cfg_p1.fft_size - csi_effective_subcarriers) - csi_guard_carriers_1

    # Resource grid for channel estimation
    rg_csi_p1 = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=cfg_p1.fft_size,
                          subcarrier_spacing=cfg_p1.subcarrier_spacing,
                          num_tx=1,
                          num_streams_per_tx=num_txs_ant_p1, # -> Ramin edit
                          cyclic_prefix_length=cfg_p1.cyclic_prefix_len,
                          num_guard_carriers=[csi_guard_carriers_1, csi_guard_carriers_2],
                          dc_null=cfg_p1.dc_null,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

    cfg_p1.num_guard_carriers = rg_csi_p1.num_guard_carriers
    #### TODO : I think I need to move the following code block to after MCS selection of phase 1
    # Channel CSI estimation using channels in previous frames/slots
    if cfg_p1.perfect_csi is True:
        # Perfect channel estimation
        h_freq_csi_dl, rx_snr_db_dl, rx_pwr_dbm_dl = p1_chans_dl.load_channel(slot_idx=cfg_p1.first_slot_idx - cfg_p1.csi_delay,
                                                                            forward=True,
                                                                            batch_size=cfg_p1.num_slots_p1)                                                                            
    # elif cfg.csi_prediction is True:
    #     rc_predictor = standard_rc_pred_freq_mimo('MU_MIMO', cfg.num_tx_streams)
    #     # Get CSI history
    #     # TODO: optimize channel estimation and optimization procedures (currently very slow)
    #     h_freq_csi_history = rc_predictor.get_csi_history(cfg.first_slot_idx, cfg.csi_delay,
    #                                                       rg_csi, dmimo_chans, 
    #                                                       cfo_vals=cfg.random_cfo_vals,
    #                                                       sto_vals=cfg.random_sto_vals)
    #     # Do channel prediction
    #     h_freq_csi = rc_predictor.rc_siso_predict(h_freq_csi_history)
    else:
        # LMMSE channel estimation
        h_freq_csi_dl, _ = lmmse_channel_estimation(p1_chans_dl, rg_csi_p1,
                                                           slot_idx=cfg_p1.first_slot_idx - cfg_p1.csi_delay,
                                                           cfo_vals=cfg_p1.random_cfo_vals,
                                                           sto_vals=cfg_p1.random_sto_vals)
        precoding_channel = h_freq_csi_dl

        _, rx_snr_db_p1, _ = p1_chans_dl.load_channel(slot_idx=cfg_p1.first_slot_idx - cfg_p1.csi_delay,
                                                                            forward=True,
                                                                            batch_size=cfg_p1.num_slots_p1)

    # print ("h_freq_dl", h_freq_csi_dl.shape)
    # print ("h_freq_ul", h_freq_csi_ul.shape)
    # # TODO: remove this later, this is just for testing 2 Tx PMI feedback
    # h_freq_csi_dl = h_freq_csi_dl[:,:,:,:,:2,...]

    if cfg_p1.CSI_feedback_method =='5G':
        generate_CSI_feedback = quantized_CSI_feedback(method='5G', codebook_selection_method='rate', num_tx_streams=cfg_p1.num_tx_streams, architecture='dMIMO_phase1', 
                                                        snrdb=rx_snr_db, wideband=True)
        [PMI, rate_for_selected_precoder, quantized_channels] = generate_CSI_feedback(h_freq_csi_dl)
    else:
        quantized_channels = None

    quantized_channels = quantized_channels[:cfg_p1.num_scheduled_tx_ue, ...]

    phase_1_enabled = getattr(cfg, "phase_1_enabled", True)

    # Create Phase 1 simulation
    mcs_list = [
        [2, 0.5],      # MCS 1: QPSK 1/2
        [2, 0.75],     # MCS 2: QPSK 3/4
        [4, 0.5],      # MCS 3: 16-QAM 1/2
        [4, 0.75],     # MCS 4: 16-QAM 3/4
        [6, 0.6667],   # MCS 5: 64-QAM 2/3
        [6, 0.75],     # MCS 6: 64-QAM 3/4
        [6, 0.8333],   # MCS 7: 64-QAM 5/6
        [8, 0.75],     # MCS 8: 256-QAM 3/4
        [8, 0.8333],   # MCS 9: 256-QAM 5/6
        [10, 0.75],    # MCS 10: 1024-QAM 3/4
        [10, 0.8333],  # MCS 11: 1024-QAM 5/6
        [12, 0.75],    # MCS 12: 4096-QAM 3/4
        [12, 0.8333]   # MCS 13: 4096-QAM 5/6
    ] # Taken from WiFi 7 (Source: Gemini)

    # The binary source will create batches of information bits
    binary_source = BinarySource()
    # Generate information bits for phase 2 (which is also the end-to-end total number of bits we want to send)
    info_bits_p2 = binary_source([cfg.num_slots_p2, mu_mimo.num_bits_per_frame])

    num_info_bits_p2 = mu_mimo.batch_size * mu_mimo.rg.num_streams_per_tx * mu_mimo.num_codewords * mu_mimo.ldpc_k

    if phase_1_enabled:
        chosen_mcs_phase_1  = None
        for mcs in mcs_list:
            cfg_p1.modulation_order  = mcs[0]
            cfg_p1.code_rate = mcs[1]
            phase_1v = Phase1v(cfg_p1, rg_csi_p1) # I need this here to have rg created before I can calculate the number of info bits for phase 1
            num_info_bits_p1 = phase_1v.batch_size * phase_1v.rg.num_streams_per_tx * phase_1v.num_codewords * phase_1v.ldpc_k 
            if num_info_bits_p1 >= num_info_bits_p2:
                chosen_mcs_phase_1 = mcs
                break
        if chosen_mcs_phase_1 is None:
            raise ValueError("No suitable MCS found for phase 1 to " \
            "accommodate the number of information bits in phase 2. Consider " \
            "increasing the time duration for phase 1 or decreasing that of phase 2.")
        
        ### End Ramin edit for phase 1
        # Check the number of info bits in phase 1:

        # flatten:
        info_bits_p1 = tf.reshape(info_bits_p2, [-1]) # size: cfg.num_slots_p2 * mu_mimo.num_bits_per_frame
        # pad: (to make sure it would be compatible with phase 1's RG)
        info_bits_p1 = tf.pad(info_bits_p1, [[0, num_info_bits_p1 - num_info_bits_p2]])
        # reshape:
        info_bits_p1 = tf.reshape(info_bits_p1, [phase_1v.batch_size, phase_1v.rg.num_streams_per_tx * phase_1v.num_codewords * phase_1v.ldpc_k ])

        ## Time for Phase 1 transmission and reception:
        detected_bits_list = phase_1v(p1_chans_dl, info_bits_p1, quantized_channels, precoding_method='grad_ascent')
        available_info_bits_list = [info_bits_p2] # The base station has the error-free information
        for i_txue in range(len(detected_bits_list)):
            detected_bits = detected_bits_list[i_txue]
            # flatten
            detected_bits = tf.reshape(detected_bits, [-1])
            # trim to the original number of bits in phase 2 (in case we padded extra bits)
            detected_bits = detected_bits[:num_info_bits_p2]
            # reshape to [num_slots_p2, num_bits_per_frame]
            detected_bits = tf.reshape(detected_bits, [cfg.num_slots_p2, mu_mimo.num_bits_per_frame])
            available_info_bits_list.append(detected_bits)
        
        # compute and print the coded BER for phase 1
        print(f"\nPhase 1 coded BER with {phase_1v.rg.num_streams_per_tx} streams, modulation order {chosen_mcs_phase_1[0]}, code rate {chosen_mcs_phase_1[1]}:")
        p1_coded_bers = [compute_ber(info_bits_p2, available_info_bits_list[i]).numpy() for i in range(len(detected_bits_list))]
        if np.mean(p1_coded_bers) != 0:
            print(p1_coded_bers)
    else:
        # Phase 2 only mode: assume all transmitters have the phase-2 bits.
        available_info_bits_list = [info_bits_p2 for _ in range(ns3cfg.num_txue_sel + 1)]

    # Saving Rx SNRs
    rx_snr_lin = 10.0 **( rx_snr_db / 10.0)
    rx_snr_lin = np.mean(rx_snr_lin, axis=(0,1, 3))
    rx_snr_lin = np.reshape(rx_snr_lin, [ns3cfg.num_rxue_sel+2, -1])
    rx_snr_lin = np.mean(rx_snr_lin, axis=-1)
    snr_dB_arr = 10*np.log10(rx_snr_lin)

    # MU-MIMO transmission (P2)
    dec_bits, uncoded_ber_phase_2, uncoded_ser, x_hat, node_wise_uncoded_ser, sinr_db_arr = mu_mimo(dmimo_chans, h_freq_csi, available_info_bits_list, snr_dB_arr)
    
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
    info_bits_p2 = tf.reshape(info_bits_p2, dec_bits.shape) # shape: [batch_size, 1, num_streams_per_tx, num_codewords, num_effective_subcarriers*num_data_ofdm_syms_per_subframe]
    coded_ber = compute_ber(info_bits_p2, dec_bits).numpy()
    coded_bler = compute_bler(info_bits_p2, dec_bits).numpy()
    # print("Uncoded BER: ", uncoded_ber_phase_2)
    print("Coded BER: ", coded_ber)
    print("BLER: ", coded_bler)

    node_wise_ber, node_wise_bler = compute_UE_wise_BER(info_bits_p2, dec_bits, cfg.ue_ranks[0], cfg.num_tx_streams)
    node_wise_acks = 1 - np.ceil(node_wise_bler)
    cfg.last_mcs_indices = mcs_indices
    cfg.last_node_wise_acks = node_wise_acks

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
        print("PHASE 3 STATS\nUNCODED BER: {}\nCODED BER: {}\nBLER: {}".format(uncoded_ber_phase_3 , coded_ber_phase_3, coded_bler_phase_3))
        # if uncoded_ber_phase_3 >= 1e-2 or coded_ber_phase_3 >= 1e-2:
        #     print("Warning: High RxSquad transmission BER")
        
        # dec_bits_phase_3 = tf.reshape(dec_bits_phase_3, [dec_bits_phase_3.shape[0], forwarding_bits.shape[1], forwarding_bits.shape[2], forwarding_bits.shape[3], forwarding_bits.shape[4]])
        # gNB_bits_phase_2 = dec_bits[:,:,:-(cfg.num_scheduled_ues * cfg.ue_ranks[0]), : , :]
        # end_to_end_dec_bits = tf.concat([gNB_bits_phase_2, dec_bits_phase_3], axis=2)

        gNB_bits_phase_2 = dec_bits[:,:,:-(cfg.num_scheduled_ues * cfg.ue_ranks[0]), : , :]
        end_to_end_dec_bits = tf.concat([gNB_bits_phase_2, dec_bits_phase_3], axis=2)

        coded_ber = compute_ber(info_bits_p2, end_to_end_dec_bits).numpy()
        coded_bler = compute_bler(info_bits_p2, end_to_end_dec_bits).numpy()

        print("Coded BER with phase 3 enabled: ", coded_ber)
        print("BLER with phase 3 enabled: ", coded_bler)

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

    return [uncoded_ber_phase_2, coded_ber], [goodbits, userbits, ratedbits_phase_2], [node_wise_goodbits_phase_2, node_wise_userbits_phase_2, node_wise_ratedbits_phase_2, ranks_out, sinr_db_arr, snr_dB_arr, PMI_feedback_bits, node_wise_bler]


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

    no_rl_throughput = None
    total_steps = None

    if cfg.csi_prediction and "deqn" in cfg.channel_prediction_method:
        data = np.load('results/channels_multiple_mu_mimo/channels_high_mobility_{}/mu_mimo_results_link_adapt_rx_UE_{}_tx_UE_{}_prediction_two_mode_pmi_quantization_True.npz'.format(cfg.drop_idx, ns3cfg.num_rxue_sel, ns3cfg.num_txue_sel))
        no_rl_throughput = data['per_step_throughput']
        assert(len(no_rl_throughput) == len(np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2))-1)
        cfg.num_transitions = len(no_rl_throughput) - 1
        total_steps = cfg.num_transitions

        cfg.curr_no_rl_throughput = no_rl_throughput[0]
    else:
        cfg.curr_no_rl_throughput = None

    if rl_selector is None and cfg.csi_prediction and "deqn" in cfg.channel_prediction_method:
        rl_selector = RLBeamSelector(total_steps=total_steps)
        checkpoint = getattr(cfg, "rl_checkpoint", None)
        if checkpoint:
            rl_selector.load_all(Path(checkpoint))
        rl_selector.set_evaluation_mode(bool(getattr(cfg, "rl_evaluation_only", False)))
    cfg.rl_selector = rl_selector

    for first_slot_idx in np.arange(cfg.start_slot_idx, cfg.total_slots, cfg.num_slots_p1 + cfg.num_slots_p2):
        
        # print("first_slot_idx: ", first_slot_idx)

        cfg.first_slot_idx = first_slot_idx

        start_time = time.time()
        bers, bits, additional_KPIs = sim_mu_mimo(cfg, ns3cfg, rc_config)
        end_time = time.time()
        # print("Cycle time: ", end_time - start_time, " seconds\n")
        
        if first_slot_idx > cfg.start_slot_idx:

            if cfg.csi_prediction and "deqn" in cfg.channel_prediction_method:
                cfg.curr_no_rl_throughput = no_rl_throughput[total_cycles]

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
        
        hold = 1

    goodput = goodput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    throughput = throughput / (total_cycles * slot_time * 1e6) * overhead  # Mbps
    bitrate = bitrate / (total_cycles * slot_time * 1e6) * overhead  # Mbps

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

    return [
        uncoded_ber / total_cycles,
        ldpc_ber / total_cycles,
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
        per_step_throughput
    ]