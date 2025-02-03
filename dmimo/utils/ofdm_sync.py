"""
OFDM frequency and timing synchronization
"""

import tensorflow as tf
import numpy as np

from sionna.signal import fft, ifft


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


def add_frequency_offset(x, cfo_vals, subcarrier_spacing=15e3, cp_len=64, channel_type="dMIMO"):
    """
    Add frequency offset errors to OFDM signals
    1) BS antennas has zero CFO errors
    2) all antennas on the same UE have the same CFO

    :param x: OFDM signal grid
    :param cfo_vals: random CFO values
    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :param cp_len: OFDM cyclic prefix length
    :return: OFDM signal grid with random frequency offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    # num_bs_ant, num_ue_ant = 4, 2  # TODO: param for BS/UE antennas
    num_total_ant = x.shape[2]  # multiple Tx support?
    # num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))
    num_ofdm_sym, fft_size = x.shape[-2:]
    num_slots = x.shape[0]  # number of slots in one Phase 2 dMIMO transmission cycle

    if channel_type == 'dMIMO':
        cfo_vals = np.repeat(cfo_vals, repeats=2, axis=0)
        cfo_vals = np.concatenate((np.zeros((1, 4, 1, 1)), np.reshape(cfo_vals, (1, -1, 1, 1))), axis=1)
    elif channel_type == 'RxSquad':
        cfo_vals = np.repeat(cfo_vals, repeats=2, axis=0)
        cfo_vals[:2, :] = 0
        cfo_vals = np.reshape(cfo_vals, (1, -1, 1, 1))
    else:
        raise Exception(f"Unsupported channel_type.")
    
    cfo_vals = normalize_cfo(subcarrier_spacing, cfo_vals[:, :num_total_ant])

    # normalized sampling time indices for multiple subframes
    time_indices = np.linspace(0, num_slots * num_ofdm_sym * (fft_size + cp_len) / fft_size,
                               num_slots * num_ofdm_sym * (fft_size + cp_len), endpoint=False)
    cfo_phase = cfo_vals * time_indices.reshape((num_slots, 1, num_ofdm_sym, fft_size + cp_len))
    cfo_phase = cfo_phase[..., cp_len:]  # remove cyclic prefix parts
    cfo_phase = np.exp(2j * np.pi * cfo_phase)
    cfo_phase = np.reshape(cfo_phase, (num_slots, 1, -1, num_ofdm_sym, fft_size))

    # convert signal to time-domain
    xt = ifft(x)
    # apply phase rotation caused by frequency offset
    xt = tf.cast(cfo_phase, tf.complex64) * xt
    # convert signal back to frequency-domain
    xf = fft(xt)

    return xf


def add_timing_offset(x, sto_vals, subcarrier_spacing=15e3, channel_type="dMIMO"):
    """
    Modeling fractional STO in frequency domain
    1) BS antennas has zero STO errors
    2) all antennas on the same UE have the same STO

    :param x: OFDM signal grid
    :param sto_vals: random STO values
    :param subcarrier_spacing: OFDM subcarrier spacing (in Hz)
    :return: OFDM signal grid with random timing offsets added
    """

    # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, num_subcarriers]
    # num_bs_ant, num_ue_ant = 4, 2  # TODO: param for BS/UE antennas
    num_total_ant = x.shape[2]  # multiple Tx support?
    # num_ue = int(np.ceil((num_total_ant - num_bs_ant) / num_ue_ant))
    num_ofdm_sym, fft_size = x.shape[-2:]

    if channel_type == 'dMIMO':
        sto_vals = np.repeat(sto_vals, repeats=2, axis=0)
        sto_vals = np.concatenate((np.zeros((4, 1, 1)), np.reshape(sto_vals, (-1, 1, 1))), axis=0)
    elif channel_type == 'RxSquad':
        sto_vals = np.repeat(sto_vals, repeats=2, axis=0)
        sto_vals[:2, :] = 0
        sto_vals = np.reshape(sto_vals, (1, -1, 1, 1))
    else:
        raise Exception(f"Unsupported channel_type.")
    sto_vals = normalize_sto(subcarrier_spacing, fft_size, sto_vals[:num_total_ant])
    # maximum relative STO magnitude is 0.5
    sto_vals[sto_vals > 0.5] = 0.5
    sto_vals[sto_vals < -0.5] = -0.5
    # compute phase shift in frequency domain which remain constant for all OFDM symbols
    sto_shift = sto_vals * np.linspace(-0.5, 0.5, fft_size, endpoint=False).reshape((1, 1, fft_size))
    phase_shift = np.exp(2j * np.pi * sto_shift)
    phase_shift = np.reshape(phase_shift, (1, 1, -1, 1, fft_size))

    # apply STO to BS/UE streams
    x = tf.cast(phase_shift, tf.complex64) * x

    return x
