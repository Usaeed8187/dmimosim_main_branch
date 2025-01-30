"""
Layer for implementing an dMIMO channels in the frequency domain,
including TxSquad, dMIMO, and RxSquad models
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Layer

from sionna.channel import ApplyOFDMChannel, AWGN
from sionna.ofdm import ResourceGrid

from dmimo.config import Ns3Config
from .ns3_channels import LoadNs3Channel


class dMIMOChannels(Layer):
    """
    dMIMOChannels apply inputs the specific type of channels and generate received output signals.
    """

    def __init__(self, config: Ns3Config, channel_type, forward=True, resource_grid: ResourceGrid = None,
                 add_noise=True, normalize_channel=False, return_channel=False, return_rxpwr=False,
                 dtype=tf.complex64, **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self._config = config
        self._channel_type = channel_type
        self._rg = resource_grid
        self._add_noise = add_noise
        self._normalize_channel = normalize_channel
        self._return_channel = return_channel
        self._return_rxpwr = return_rxpwr
        self._load_channel = LoadNs3Channel(self._config)
        self._apply_channel = ApplyOFDMChannel(add_awgn=False, dtype=tf.as_dtype(self.dtype))
        self._awgn = AWGN(dtype=dtype)
        self._forward = forward

    @property
    def ns3_config(self):
        return self._config

    @property
    def ns3_channel(self):
        return self._load_channel

    @property
    def channel_type(self):
        return self._channel_type

    @property
    def forward(self):
        return self._forward

    def load_channel(self, slot_idx, forward=forward, batch_size=1, ue_selection=True):
        assert slot_idx >= 0, "Slot indices must be non-negative integers"
        return self._load_channel(self._channel_type, forward=forward, slot_idx=slot_idx, batch_size=batch_size,
                                  ue_selection=ue_selection)

    def call(self, inputs):

        # x: channel input samples, sidx: current slot index
        if len(inputs) == 2:
            x, sidx = inputs
            tx_mask, rx_mask = None, None
        else:
            x, sidx, tx_mask, rx_mask = inputs

        # x has shape [batch_size, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        batch_size = tf.shape(x)[0]
        total_tx_ant = tf.shape(x)[1] * tf.shape(x)[2]
        # num_txs_ant = self._config.num_bs * self._config.num_bs_ant + self._config.num_txue * self._config.num_ue_ant
        # assert num_txs_ant == total_tx_ant, "Total number of transmit antennas of input and channel must match"

        # load pre-generated ns-3 channels
        # h_freq shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym, fft_size]
        # rx_snr_db shape: [batch_size, 1, num_rx_ant, num_ofdm_sym]
        h_freq, rx_snr_db, rx_pwr_dbm = self._load_channel(self._channel_type, slot_idx=sidx, batch_size=batch_size)

        # Prune data and channel subcarriers according to the resource grid
        if self._rg and x.shape[-1] != h_freq.shape[-1]:
            assert self._rg.num_effective_subcarriers <= x.shape[-1]
            assert self._rg.num_effective_subcarriers <= h_freq.shape[-1]
            scidx = self._rg.effective_subcarrier_ind
            if x.shape[-1] != self._rg.num_effective_subcarriers:
                x = tf.gather(x, scidx, axis=-1)
            if h_freq.shape[-1] != self._rg.num_effective_subcarriers:
                h_freq = tf.gather(h_freq, scidx, axis=-1)

        # Apply channel to inputs
        if tx_mask is None and rx_mask is None:
            y = self._apply_channel([x, h_freq])  # [batch_size, num_rx, num_rx_ant, num_ofdm_sym, fft_size]
        else:
            if tx_mask is not None:
                h_freq = tf.gather(h_freq, tx_mask, axis=4)
                y = self._apply_channel([x, h_freq])
            if rx_mask is not None:
                y = tf.gather(y, rx_mask, axis=1)

        # Add thermal noise
        if self._add_noise:
            no = tf.cast(np.power(10.0, rx_snr_db / (-10.0)), tf.float32)
            no = tf.expand_dims(no, -1)  # [batch_size, num_rx, num_rx_ant, num_ofdm_sym, 1]
            y = self._awgn([y, no])

        if self._return_channel:
            return y, h_freq
        elif self._return_rxpwr:
            return y, rx_pwr_dbm
        else:
            return y, None
