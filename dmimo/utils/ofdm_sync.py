"""
OFDM frequency and timing synchronization
"""

import numpy as np
import tensorflow as tf


def normalize_cfo(subcarrier_spacing, cfo_hz):
    """
    Compute CFO value relative to subcarrier spacing

    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :param cfo_hz: CFO standard deviation (in Hz)
    :return: normalized CFO standard deviation
    """
    return cfo_hz / subcarrier_spacing


def normalize_sto(subcarrier_spacing, fft_size, sto_ns):
    """
    Compute STO value relative to baseband sample duration

    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :param fft_size: OFDM FFT size
    :param sto_ns: STO standard deviation (in nanosecond)
    :return: normalized STO standard deviation
    """
    ts = 1.0 / (subcarrier_spacing * fft_size)
    return (sto_ns * 1e-9) / ts


def add_frequency_offset(
    x,
    cfo_vals,
    subcarrier_spacing=15e3,
    cp_len=64,
    channel_type="dMIMO",
    slot_idx=0,
    slot_duration=1e-3,
):
    """
    Add the per-slot common phase caused by residual frequency offset.
    1) BS antennas has zero CFO errors
    2) all antennas on the same UE have the same CFO

    :param x: OFDM signal grid
    :param cfo_vals: random CFO values
    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :param cp_len: retained for API compatibility
    :param slot_idx: absolute ns-3 slot index (slot zero is the phase origin)
    :param slot_duration: slot duration in seconds
    :return: OFDM signal grid with random frequency offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    # num_bs_ant, num_ue_ant = 4, 2  # TODO: param for BS/UE antennas
    num_total_ant = x.shape[2]  # multiple Tx support?
    # num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))
    if channel_type == 'dMIMO':
        cfo_vals = np.repeat(cfo_vals, repeats=2, axis=0)
        cfo_vals = np.concatenate((np.zeros((1, 4, 1, 1)), np.reshape(cfo_vals, (1, -1, 1, 1))), axis=1)
    elif channel_type == 'RxSquad':
        cfo_vals = np.repeat(cfo_vals, repeats=2, axis=0)
        cfo_vals[:2, :] = 0
        cfo_vals = np.reshape(cfo_vals, (1, -1, 1, 1))
    else:
        raise Exception(f"Unsupported channel_type.")
    
    cfo_vals = np.asarray(cfo_vals[:, :num_total_ant], dtype=np.float64)
    phase = np.exp(
        2j * np.pi * float(slot_idx) * float(slot_duration)
        * cfo_vals.reshape((1, 1, -1, 1, 1))
    )
    return tf.cast(phase, x.dtype) * x


def add_timing_offset(
    x,
    sto_vals,
    subcarrier_spacing=15e3,
    cp_len=64,
    channel_type="dMIMO",
):
    """
    Modeling fractional STO in frequency domain
    1) BS antennas has zero STO errors
    2) all antennas on the same UE have the same STO

    :param x: OFDM signal grid
    :param sto_vals: random STO values
    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :param cp_len: OFDM cyclic prefix length in samples
    :return: OFDM signal grid with random timing offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    # num_bs_ant, num_ue_ant = 4, 2  # TODO: param for BS/UE antennas
    num_total_ant = x.shape[2]  # multiple Tx support?
    # num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))
    fft_size = int(x.shape[-1])

    if channel_type == 'dMIMO':
        sto_vals = np.repeat(sto_vals, repeats=2, axis=0)
        sto_vals = np.concatenate((np.zeros((4, 1, 1)), np.reshape(sto_vals, (-1, 1, 1))), axis=0)
    elif channel_type == 'RxSquad':
        sto_vals = np.repeat(sto_vals, repeats=2, axis=0)
        sto_vals[:2, :] = 0
        sto_vals = np.reshape(sto_vals, (1, -1, 1, 1))
    else:
        raise Exception(f"Unsupported channel_type.")
    sto_vals = normalize_sto(
        subcarrier_spacing, fft_size, np.asarray(sto_vals[:num_total_ant])
    )
    if np.any(np.abs(sto_vals) >= cp_len):
        raise ValueError(
            "Residual timing offsets must lie strictly within the cyclic prefix "
            f"of {cp_len} samples."
        )
    # The resource grid uses FFT-shifted subcarrier order. A delay eta samples
    # therefore contributes exp(-j 2 pi k eta/N) on subcarrier k.
    subcarrier_indices = np.arange(-fft_size // 2, fft_size // 2)
    sto_shift = (
        sto_vals
        * subcarrier_indices.reshape((1, 1, fft_size))
        / float(fft_size)
    )
    phase_shift = np.exp(-2j * np.pi * sto_shift)
    phase_shift = np.reshape(phase_shift, (1, 1, -1, 1, fft_size))

    # apply STO to BS/UE streams
    x = tf.cast(phase_shift, tf.complex64) * x

    return x
