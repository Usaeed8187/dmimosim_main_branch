"""
Channel estimation for dMIMO scenarios
"""

import numpy as np
import tensorflow as tf
import time

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator
from sionna.mapping import Mapper
from sionna.utils import BinarySource, ebnodb2no

from dmimo.utils import add_frequency_offset, add_timing_offset
from .dmimo_channels import dMIMOChannels
from .interpolation import LMMSELinearInterp


def estimate_freq_cov(dmimo_chans: dMIMOChannels, rg: ResourceGrid, start_slot, total_slots=5):

    num_sc = tf.cast(rg.num_effective_subcarriers, tf.int64)
    freq_cov_mat = tf.zeros([num_sc, num_sc], tf.complex64)
    trim_h_freq = True if rg.num_effective_subcarriers != rg.fft_size else False

    for slot_idx in np.arange(start_slot, start_slot+total_slots, 1):
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq, snrdb, rxpwr = dmimo_chans.load_channel(slot_idx=slot_idx)
        # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq = np.squeeze(h_freq, axis=(1, 3))
        if trim_h_freq:
            h_freq = h_freq[..., rg.effective_subcarrier_ind]

        # [batch_size, num_tx_ant, num_rx_ant, num_ofdm_symbols, num_sc]
        h_samples = tf.transpose(h_freq, (0, 2, 1, 3, 4))

        # [num_batch, num_tx_ant, num_rx_ant, num_sc, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 2, 4, 3])
        # [num_tx_ant, num_rx_ant, num_sc, num_sc]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_sc, num_sc]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1, 2))
        # [num_sc, num_sc]
        freq_cov_mat += freq_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(total_slots, tf.float32), tf.cast(0.0, tf.float32))

    return freq_cov_mat


def estimate_freq_time_cov(dmimo_chans: dMIMOChannels, rg: ResourceGrid, start_slot, total_slots=5):

    num_sc = tf.cast(rg.num_effective_subcarriers, tf.int64)
    num_ofdm_syms = tf.cast(rg.num_ofdm_symbols, tf.int64)
    freq_cov_mat = tf.zeros([num_sc, num_sc], tf.complex64)
    time_cov_mat = tf.zeros([num_ofdm_syms, num_ofdm_syms], tf.complex64)
    trim_h_freq = True if rg.num_effective_subcarriers != rg.fft_size else False

    for slot_idx in np.arange(start_slot, start_slot+total_slots, 1):
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq, snrdb, rxpwr = dmimo_chans.load_channel(slot_idx=slot_idx)
        # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq = np.squeeze(h_freq, axis=(1, 3))
        if trim_h_freq:
            h_freq = h_freq[..., rg.effective_subcarrier_ind]

        # [batch_size, num_tx_ant, num_rx_ant, num_ofdm_symbols, num_sc]
        h_samples = tf.transpose(h_freq, (0, 2, 1, 3, 4))

        # [num_batch, num_tx_ant, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 2, 4, 3])
        # [num_tx_ant, num_rx_ant, num_sc, num_sc]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_sc, num_sc]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1, 2))
        # [num_sc, num_sc]
        freq_cov_mat += freq_cov_mat_

        # [batch size, num_rx_ant, num_ofdm_symbols, num_sc]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0, 1, 2))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(total_slots, tf.float32), tf.cast(0.0, tf.float32))
    time_cov_mat /= tf.complex(tf.cast(num_sc*total_slots, tf.float32), 0.0)

    return freq_cov_mat, time_cov_mat


def lmmse_channel_estimation(dmimo_chans: dMIMOChannels, rg: ResourceGrid, slot_idx, cache_slots=5, ebno_db=12.0,
                             cfo_vals=[0], sto_vals=[0], freq_cov_mat=None, lmmse_interpolator=None,
                             use_rx_snr_for_nvar=True):
    # Only allow channel estimation from slot 1 onward
    assert slot_idx >= 0, "Current slot index must be a positive integer"

    num_bits_per_symbol = 2  # use QPSK modulation
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(rg)
    
    start_time = time.time()
    if lmmse_interpolator is None:
        # Make sure slot_idx is always non-negative
        if slot_idx - cache_slots < 0:
            cache_slots = slot_idx
        start_slot = slot_idx - cache_slots + 1
        if freq_cov_mat is None:
            freq_cov_mat = estimate_freq_cov(dmimo_chans, rg, start_slot=start_slot, total_slots=cache_slots)
        lmmse_int = LMMSELinearInterp(rg.pilot_pattern, freq_cov_mat)
    else:
        lmmse_int = lmmse_interpolator
    end_time = time.time()
    # print("Time taken for LMMSELinearInterp intitialization: ", end_time - start_time)
    
    ls_estimator = LSChannelEstimator(rg, interpolator=lmmse_int)

    # Calculate noise variance for LS channel estimation.
    # Optionally derive it from the per-slot NS-3 Rx SNR so channel-estimation
    # quality tracks transmit power / pathloss changes.
    if use_rx_snr_for_nvar:
        _, rx_snr_db, _ = dmimo_chans.load_channel(slot_idx=slot_idx, batch_size=1)
        rx_snr_lin = np.power(10.0, np.asarray(rx_snr_db) / 10.0)
        nvar = tf.cast(np.mean(1.0 / np.maximum(rx_snr_lin, 1e-12)), tf.float32)
    else:
        nvar = ebnodb2no(ebno_db, num_bits_per_symbol, 0.5)

    # Generate OFDM grid signals
    bs = binary_source([1, 1, rg.num_streams_per_tx, rg.num_data_symbols * num_bits_per_symbol])
    dx = mapper(bs)
    dx_rg = rg_mapper(dx)

    # add CFO/STO to simulate synchronization errors
    if np.any(np.not_equal(sto_vals, 0)):
        dx_rg = add_timing_offset(
            dx_rg,
            sto_vals,
            subcarrier_spacing=rg.subcarrier_spacing,
            cp_len=rg.cyclic_prefix_length,
            channel_type=dmimo_chans.channel_type,
        )
    if np.any(np.not_equal(cfo_vals, 0)):
        dx_rg = add_frequency_offset(
            dx_rg,
            cfo_vals,
            subcarrier_spacing=rg.subcarrier_spacing,
            cp_len=rg.cyclic_prefix_length,
            channel_type=dmimo_chans.channel_type,
            slot_idx=slot_idx,
        )

    # Pass through ns3 channels
    # output has shape: [1, num_rx, num_rx_ant, num_ofdm_sym, fft_size]
    ry, _ = dmimo_chans([dx_rg, slot_idx])

    #
    # LMMSE channel estimation
    #
    num_rx_ant = ry.shape[2]
    h_all = []
    err_var_all = []
    # loop for individual receiver antennas in each batch to reduce memory requirement
    for idx in range(num_rx_ant):
        ry1 = ry[:1, :1, idx:idx+1, :, :]
        h_hat, err_var = ls_estimator([ry1, nvar])
        h_all.append(h_hat)
        err_var_all.append(err_var)

    h_all = tf.concat(h_all, axis=2)
    evar_all = tf.concat(err_var_all, axis=2)

    # Guard subcarriers padding
    if np.sum(rg.num_guard_carriers) > 0:
        h_freq_guard1 = tf.gather(h_all, np.repeat(0, rg.num_guard_carriers[0]), axis=-1)
        h_freq_guard2 = tf.gather(h_all, np.repeat(rg.num_effective_subcarriers-1, rg.num_guard_carriers[1]), axis=-1)
        h_all = tf.concat((h_freq_guard1, h_all, h_freq_guard2), axis=-1)

    return h_all, evar_all

def _trim_to_effective_subcarriers(ry, rg: ResourceGrid):
    """Keep only effective subcarriers from a received OFDM grid."""

    if rg.num_effective_subcarriers == rg.fft_size:
        return ry
    return tf.gather(ry, rg.effective_subcarrier_ind, axis=-1)


def get_received_pilot_symbols(dmimo_chans: dMIMOChannels, rg: ResourceGrid, slot_idx,
                               cfo_vals=[0], sto_vals=[0], num_bits_per_symbol=2):
    """
    Simulate one slot and return received samples on pilot OFDM symbols.

    Returns
    -------
    ry_pilot_eff : tf.Tensor
        Shape [batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]
    pilot_symbols : tf.Tensor
        Shape [num_tx=1, num_streams_per_tx, num_pilot_symbols]
    """

    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(rg)

    bs = binary_source([1, 1, rg.num_streams_per_tx, rg.num_data_symbols * num_bits_per_symbol])
    dx = mapper(bs)
    dx_rg = rg_mapper(dx)

    if np.any(np.not_equal(sto_vals, 0)):
        dx_rg = add_timing_offset(
            dx_rg,
            sto_vals,
            subcarrier_spacing=rg.subcarrier_spacing,
            cp_len=rg.cyclic_prefix_length,
            channel_type=dmimo_chans.channel_type,
        )
    if np.any(np.not_equal(cfo_vals, 0)):
        dx_rg = add_frequency_offset(
            dx_rg,
            cfo_vals,
            subcarrier_spacing=rg.subcarrier_spacing,
            cp_len=rg.cyclic_prefix_length,
            channel_type=dmimo_chans.channel_type,
            slot_idx=slot_idx,
        )

    ry, _ = dmimo_chans([dx_rg, slot_idx])
    ry_eff = _trim_to_effective_subcarriers(ry, rg)
    ry_pilot_eff = tf.gather(ry_eff, rg._pilot_ofdm_symbol_indices, axis=-2)
    pilot_symbols = tf.convert_to_tensor(np.asarray(rg.pilot_pattern.pilots))

    return ry_pilot_eff, pilot_symbols


def _extract_pilot_values_from_rx_symbols(ry_pilot_eff, rg: ResourceGrid):
    """
    Convert received pilot-OFDM-symbol grids to per-stream pilot vectors.

    Parameters
    ----------
    ry_pilot_eff : tf.Tensor
        Shape [batch, num_rx, num_rx_ant, num_pilot_ofdm_symbols, num_effective_subcarriers]

    Returns
    -------
    y_p : tf.Tensor
        Shape [batch, num_rx, num_rx_ant, num_tx=1, num_streams_per_tx, num_pilot_symbols]
    """

    ry_pilot_eff = tf.convert_to_tensor(ry_pilot_eff)

    # [num_streams, num_pilot_ofdm_symbols, num_effective_subcarriers]
    pilot_mask = tf.convert_to_tensor(np.asarray(rg.pilot_pattern.mask)[0], dtype=tf.bool)
    pilot_mask = tf.gather(pilot_mask, rg._pilot_ofdm_symbol_indices, axis=1)
    num_streams = pilot_mask.shape[0]

    # Flatten OFDM-symbol and subcarrier dimensions so the pilot mask can index REs
    # consistently across all batch/rx dimensions.
    ry_flat = tf.reshape(ry_pilot_eff, [*ry_pilot_eff.shape[:3], -1])
    mask_flat = tf.reshape(pilot_mask, [num_streams, -1])

    y_list = []
    for stream_idx in range(num_streams):
        pilot_idx = tf.where(mask_flat[stream_idx])[:, 0]
        y_stream = tf.gather(ry_flat, pilot_idx, axis=-1)
        y_list.append(y_stream[:, :, :, tf.newaxis, tf.newaxis, :])

    return tf.concat(y_list, axis=4)

def estimate_channel_from_pilot_rx_symbols(ry_pilot_eff, rg: ResourceGrid, pilot_symbols,
                                           ebno_db=12.0, freq_cov_mat=None, lmmse_interpolator=None):
    """
    Estimate full-grid channel from predicted/observed pilot-domain received signal.

    Steps:
      1) LS on pilot REs: h_ls_p = y_p / x_p
      2) LMMSE interpolation to full grid
    """

    if lmmse_interpolator is None:
        if freq_cov_mat is None:
            raise ValueError("freq_cov_mat is required when lmmse_interpolator is not provided.")
        lmmse_int = LMMSELinearInterp(rg.pilot_pattern, freq_cov_mat)
    else:
        lmmse_int = lmmse_interpolator

    y_p = _extract_pilot_values_from_rx_symbols(ry_pilot_eff, rg)
    x_p = tf.convert_to_tensor(pilot_symbols, dtype=y_p.dtype)  # [num_tx=1, num_streams, num_pilots]
    x_p = x_p[tf.newaxis, tf.newaxis, tf.newaxis, ...]  # broadcast dims

    h_ls_p = tf.math.divide_no_nan(y_p, x_p)

    nvar = ebnodb2no(ebno_db, 2, 0.5)
    err_var_p = tf.ones(tf.shape(h_ls_p), dtype=tf.float32) * tf.cast(nvar, tf.float32)

    h_eff, err_eff = lmmse_int(h_ls_p, err_var_p)

    if np.sum(rg.num_guard_carriers) > 0:
        h_freq_guard1 = tf.gather(h_eff, np.repeat(0, rg.num_guard_carriers[0]), axis=-1)
        h_freq_guard2 = tf.gather(h_eff, np.repeat(rg.num_effective_subcarriers - 1, rg.num_guard_carriers[1]), axis=-1)
        h_eff = tf.concat((h_freq_guard1, h_eff, h_freq_guard2), axis=-1)

        err_guard1 = tf.gather(err_eff, np.repeat(0, rg.num_guard_carriers[0]), axis=-1)
        err_guard2 = tf.gather(err_eff, np.repeat(rg.num_effective_subcarriers - 1, rg.num_guard_carriers[1]), axis=-1)
        err_eff = tf.concat((err_guard1, err_eff, err_guard2), axis=-1)

    return h_eff, err_eff

def lmmse_channel_estimation_demo(ry, rg: ResourceGrid, slot_idx, cache_slots=5, ebno_db=12.0,
                             cfo_vals=[0], sto_vals=[0]):

    # Only allow channel estimation from slot 1 onward
    assert slot_idx > 0, "Current slot index must be a positive integer"

    # Make sure slot_idx is always non-negative
    if slot_idx - cache_slots < 0:
        cache_slots = slot_idx
    start_slot = slot_idx - cache_slots + 1

    num_bits_per_symbol = 2  # use QPSK modulation
    binary_source = BinarySource()
    mapper = Mapper("qam", num_bits_per_symbol)
    rg_mapper = ResourceGridMapper(rg)

    freq_cov_mat = estimate_freq_cov_demo(rg, start_slot=start_slot, total_slots=cache_slots)
    lmmse_int = LMMSELinearInterp(rg.pilot_pattern, freq_cov_mat)
    ls_estimator = LSChannelEstimator(rg, interpolator=lmmse_int)

    # Calculate noise variance for LS channel estimation
    nvar = ebnodb2no(ebno_db, num_bits_per_symbol, 0.5)

    #
    # LMMSE channel estimation
    #
    num_rx_ant = ry.shape[2]
    h_all = []
    err_var_all = []
    # loop for individual receiver antennas in each batch to reduce memory requirement
    for idx in range(num_rx_ant):
        ry1 = ry[:1, :1, idx:idx+1, :, :]
        h_hat, err_var = ls_estimator([ry1, nvar])
        h_all.append(h_hat)
        err_var_all.append(err_var)

    h_all = tf.concat(h_all, axis=2)
    evar_all = tf.concat(err_var_all, axis=2)

    # Guard subcarriers padding
    if np.sum(rg.num_guard_carriers) > 0:
        h_freq_guard1 = tf.gather(h_all, np.repeat(0, rg.num_guard_carriers[0]), axis=-1)
        h_freq_guard2 = tf.gather(h_all, np.repeat(rg.num_effective_subcarriers-1, rg.num_guard_carriers[1]), axis=-1)
        h_all = tf.concat((h_freq_guard1, h_all, h_freq_guard2), axis=-1)

    return h_all, evar_all

def estimate_freq_cov_demo(rg: ResourceGrid, start_slot, total_slots=5):

    num_sc = tf.cast(rg.num_effective_subcarriers, tf.int64)
    freq_cov_mat = tf.zeros([num_sc, num_sc], tf.complex64)
    trim_h_freq = True if rg.num_effective_subcarriers != rg.fft_size else False

    for slot_idx in np.arange(start_slot, start_slot+total_slots, 1):
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq, snrdb, rxpwr = dmimo_chans.load_channel(slot_idx=slot_idx)
        # [batch_size, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq = np.squeeze(h_freq, axis=(1, 3))
        if trim_h_freq:
            h_freq = h_freq[..., rg.effective_subcarrier_ind]

        # [batch_size, num_tx_ant, num_rx_ant, num_ofdm_symbols, num_sc]
        h_samples = tf.transpose(h_freq, (0, 2, 1, 3, 4))

        # [num_batch, num_tx_ant, num_rx_ant, num_sc, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0, 1, 2, 4, 3])
        # [num_tx_ant, num_rx_ant, num_sc, num_sc]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_sc, num_sc]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0, 1, 2))
        # [num_sc, num_sc]
        freq_cov_mat += freq_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(total_slots, tf.float32), tf.cast(0.0, tf.float32))

    return freq_cov_mat
