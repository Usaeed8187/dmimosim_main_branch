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
from dmimo.channel import standard_rc_pred_freq_mimo, default_ddpg_predictor
from dmimo.channel.ddpg_predictor import DDPGChannelPredictor
from dmimo.channel import weiner_filter_pred
from dmimo.channel.twomode_wesn_pred import predict_all_links, predict_all_links_simple
from dmimo.channel.twomode_wesn_pred_real import predict_all_links_real, predict_all_links_simple_real
from dmimo.channel.rl_beam_selector_v2 import RLBeamSelector
from dmimo.channel.twomode_wesn_pred_tf import predict_all_links_tf
from dmimo.channel import RBwiseLinearInterp
from dmimo.mimo import BDPrecoder, BDEqualizer, ZFPrecoder, SLNRPrecoder, QuantizedSLNRPrecoder, SLNREqualizer, QuantizedZFPrecoder, QuantizedDirectPrecoder
from dmimo.mimo import rankAdaptation, linkAdaptation
from dmimo.mimo import MUMIMOScheduler
from dmimo.mimo import update_node_selection, quantized_CSI_feedback, RandomVectorQuantizer, RandomVectorQuantizerNumpy
from dmimo.utils import add_frequency_offset, add_timing_offset, compute_UE_wise_BER, compute_UE_wise_SER, complex_pinv

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
        
        start_time_all_loops = time.time()

        h_freq_csi_up_to_date, _ = lmmse_channel_estimation(dmimo_chans, rg_csi,
                                                    slot_idx=cfg.first_slot_idx,
                                                    cfo_vals=cfg.random_cfo_vals,
                                                    sto_vals=cfg.random_sto_vals,
                                                    freq_cov_mat=freq_cov_mat,
                                                    lmmse_interpolator=lmmse_interpolator)

        h_freq_csi = predict_all_links(h_freq_csi_history, rc_config, ns3cfg, max_workers=8)
        # h_freq_csi = predict_all_links_simple(h_freq_csi_history, rc_config, ns3cfg)

        nmse_outdated = tf.reduce_mean(tf.abs(h_freq_csi_history[-1, ...] - h_freq_csi_up_to_date) ** 2) / tf.reduce_mean(tf.abs(h_freq_csi_up_to_date) ** 2)
        nmse_pred = tf.reduce_mean(tf.abs(h_freq_csi - h_freq_csi_up_to_date) ** 2) / tf.reduce_mean(tf.abs(h_freq_csi_up_to_date) ** 2)
        print(f"NMSE of outdated CSI: {nmse_outdated:.4f}, NMSE of predicted CSI: {nmse_pred:.4f}")
        
        h_freq_csi_history_abs = tf.abs(h_freq_csi_history)
        h_freq_csi_up_to_date_abs = tf.abs(h_freq_csi_up_to_date)
        h_freq_csi_abs = predict_all_links_simple_real(h_freq_csi_history_abs, rc_config, ns3cfg)
        nmse_outdated_abs = tf.reduce_mean((h_freq_csi_history_abs[-1, ...] - h_freq_csi_up_to_date_abs) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_abs) ** 2)
        nmse_pred_abs = tf.reduce_mean((h_freq_csi_abs - h_freq_csi_up_to_date_abs) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_abs) ** 2)
        print(f"NMSE of outdated CSI (Abs): {nmse_outdated_abs:.4f}, NMSE of predicted CSI (Abs): {nmse_pred_abs:.4f}")

        h_freq_csi_history_phase = tf.math.angle(h_freq_csi_history)
        h_freq_csi_up_to_date_phase = tf.math.angle(h_freq_csi_up_to_date)
        h_freq_csi_phase = predict_all_links_simple_real(h_freq_csi_history_phase, rc_config, ns3cfg)
        nmse_outdated_phase = tf.reduce_mean((h_freq_csi_history_phase[-1, ...] - h_freq_csi_up_to_date_phase) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_phase) ** 2)
        nmse_pred_phase = tf.reduce_mean((h_freq_csi_phase - h_freq_csi_up_to_date_phase) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_phase) ** 2)
        print(f"NMSE of outdated CSI (Phase): {nmse_outdated_phase:.4f}, NMSE of predicted CSI (Phase): {nmse_pred_phase:.4f}")

        h_freq_csi_history_real = tf.math.real(h_freq_csi_history)
        h_freq_csi_up_to_date_real = tf.math.real(h_freq_csi_up_to_date)
        h_freq_csi_real = predict_all_links_simple_real(h_freq_csi_history_real, rc_config, ns3cfg)
        nmse_outdated_real = tf.reduce_mean((h_freq_csi_history_real[-1, ...] - h_freq_csi_up_to_date_real) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_real) ** 2)
        nmse_pred_real = tf.reduce_mean((h_freq_csi_real - h_freq_csi_up_to_date_real) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_real) ** 2)
        print(f"NMSE of outdated CSI (Real): {nmse_outdated_real:.4f}, NMSE of predicted CSI (Real): {nmse_pred_real:.4f}")

        h_freq_csi_history_imag = tf.math.imag(h_freq_csi_history)
        h_freq_csi_up_to_date_imag = tf.math.imag(h_freq_csi_up_to_date)
        h_freq_csi_imag = predict_all_links_simple_real(h_freq_csi_history_imag, rc_config, ns3cfg)
        nmse_outdated_imag = tf.reduce_mean((h_freq_csi_history_imag[-1, ...] - h_freq_csi_up_to_date_imag) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_imag) ** 2)
        nmse_pred_imag = tf.reduce_mean((h_freq_csi_imag - h_freq_csi_up_to_date_imag) ** 2) / tf.reduce_mean((h_freq_csi_up_to_date_imag) ** 2)
        print(f"NMSE of outdated CSI (Imag): {nmse_outdated_imag:.4f}, NMSE of predicted CSI (Imag): {nmse_pred_imag:.4f}")

        end_time_all_loops = time.time()
    

    return nmse_pred_real



if __name__ == "__main__":

    rc_config = RCConfig()
    rc_config.enable_window = True
    rc_config.window_length = 3
    rc_config.num_neurons = 16
    rc_config.history_len = 8

    total_cycles = 0
    chan_pred_nmse = []

    no_rl_throughput = None
    total_steps = None

    data = np.load("H_real_imag_1kmph.npz")
    H_real_imag = data["H_real_imag"]

    hold = 1