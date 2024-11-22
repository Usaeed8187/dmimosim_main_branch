import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from dmimo.mimo import ZFPrecoder

from .fiveG_precoding import baseline_fiveG_precoder, dMIMO_p1_fiveG_max_min_precoder, dMIMO_p1_fiveG_max_min_demo_precoder


class fiveGPrecoder(Layer):
    """5G Precoder for Baseline and """

    def __init__(self,
                 rg, 
                 sm,
                 architecture,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)

        self.rg = rg
        self.sm = sm

        self.architecture = architecture
        self.num_BS_Ant = 4
        self.num_UE_Ant = 2

    def call(self, inputs):
        
        if len(inputs) == 3:
            x, precoding_matrices, self.precoding_method = inputs
        elif len(inputs) == 6:
            x, precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm, self.precoding_method = inputs
        elif len(inputs) == 7:
            x, precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm, self.precoding_method, codebook = inputs
        else:
            ValueError("calling 5G precoder with incorrect params")

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # precoding_matrices has shape
        # [num_rx, fft_size, num_tx_ant, num_rx_ant]

        # Transformations to bring precoding_matrices and x in the desired shapes

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
        x_precoded = x_precoded[..., np.newaxis]

        # Transpose precoding_matrices:
        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]
        precoding_matrices = precoding_matrices[np.newaxis, :, np.newaxis, :, :, :]
        precoding_matrices = np.repeat(precoding_matrices, x_precoded.shape[2], axis=2)
        precoding_matrices = np.repeat(precoding_matrices, x_precoded.shape[0], axis=0)
        precoding_matrices = tf.cast(precoding_matrices, dtype=x_precoded.dtype)

        # Precoding
        if self.architecture == 'baseline' and self.precoding_method == '5G_ZF_no_channel_reconstruction':
            x_precoded = baseline_fiveG_precoder(x_precoded, precoding_matrices)
        elif self.architecture == 'baseline' and self.precoding_method == '5G_ZF':
            h_freq_csi_reconstructed = self.reconstruct_channel(precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm)
            zf_precoder = ZFPrecoder(self.rg, self.sm, return_effective_channel=False)
            x_precoded = zf_precoder([x, h_freq_csi_reconstructed])
        elif self.architecture == 'dMIMO_phase1' and self.precoding_method == '5G_max_min':
            h_freq_csi_reconstructed = self.reconstruct_channel(precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm)
            x_precoded = dMIMO_p1_fiveG_max_min_precoder(x_precoded, h_freq_csi_reconstructed)
        elif self.architecture == 'dMIMO_phase1' and self.precoding_method == '5G_max_min_demo':
            _, s = self.reconstruct_channel(precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm, return_s=True)
            x_precoded = dMIMO_p1_fiveG_max_min_demo_precoder(x_precoded, precoding_matrices, s, codebook, n_var)

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        if self.precoding_method != '5G_ZF':
            x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        return x_precoded

    def reconstruct_channel(self, precoding_matrices, cqi_snr, n_var, bs_txpwr_dbm, return_s=False):
        

        # precoding_matrices has shape:
        # [batch_size, num_rx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]

        rx_sig_pow = n_var * 10**(cqi_snr/10)
        tx_sig_pow = 10**(bs_txpwr_dbm/10)
        s = np.sqrt(rx_sig_pow / tx_sig_pow)

        h_freq_csi_reconstructed = precoding_matrices * s

        if len(h_freq_csi_reconstructed.shape) == 6:
            h_freq_csi_reconstructed = tf.transpose(h_freq_csi_reconstructed, [0, 1, 5, 4, 2, 3])
            h_freq_csi_reconstructed = h_freq_csi_reconstructed[:, :, :, np.newaxis, ...]
        elif len(h_freq_csi_reconstructed.shape) == 7:
            h_freq_csi_reconstructed = tf.transpose(h_freq_csi_reconstructed, [0, 1, 6, 3, 5, 2, 4])
        else:
            raise Exception(f"Unsupported input dimensions.")

        if self.architecture == 'baseline':
            padding = self.num_BS_Ant - h_freq_csi_reconstructed.shape[2]
        elif self.architecture == 'dMIMO_phase1':
            padding = self.num_UE_Ant - h_freq_csi_reconstructed.shape[2]
        else:
            padding = 0

        padding_mask = [
            [0, 0],  # No padding on the 1st dimension
            [0, 0],  # No padding on the 2nd dimension
            [0, padding],  # Pad the 3rd dimension from 2 to 4 (2 zeros after)
            [0, 0],  # No padding on the 4th dimension
            [0, 0],  # No padding on the 5th dimension
            [0, 0],  # No padding on the 6th dimension
            [0, 0],  # No padding on the 7th dimension
        ]

        # Output shape:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        h_freq_csi_reconstructed = tf.pad(h_freq_csi_reconstructed, padding_mask)

        if return_s:
            return h_freq_csi_reconstructed, s
        else:
            return h_freq_csi_reconstructed

        