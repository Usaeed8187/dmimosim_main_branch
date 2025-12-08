import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .phase_3_mu_mimo_uplink_precoding import phase_3_mu_mimo_uplink_precoding


class phase_3_mu_mimo_uplink_precoder(Layer):

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
        
        if len(inputs) == 7:
            x, x_rg_placeholder, precoding_matrices, rg, precoding_method, MU_MIMO_RG_populated, num_subcarriers_per_RB = inputs
        else:
            ValueError("calling phase 3 uplink precoder with incorrect params")

        # x has shape
        # [batch_size, num_UE, Nsyms_per_UE]
        #
        # precoding_matrices has shape
        # [num_UE, num_ofdm_syms, fft_size, num_tx_ant, num_streams]

        # Transformations to bring precoding_matrices and x in the desired shapes

        # Transpose x:
        # [batch_size, num_UE, Nsyms_per_UE, num_streams]
        x_precoded = tf.transpose(x, [0, 1, 3, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
        x_precoded = x_precoded[..., np.newaxis]

        # Transpose x_rg_placeholder:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_rg_placeholder_precoded = tf.transpose(x_rg_placeholder, [0, 1, 3, 4, 2])
        x_rg_placeholder_precoded = tf.cast(x_rg_placeholder_precoded, self._dtype)
        x_rg_placeholder_precoded = x_rg_placeholder_precoded[..., np.newaxis]

        # Transpose precoding_matrices:
        # [num_UE, num_ofdm_syms, fft_size, num_tx_ant, num_streams]
        precoding_matrices = tf.cast(precoding_matrices, dtype=x_precoded.dtype)

        # Precoding
        if self.architecture == 'phase_3' and precoding_method == 'mu_mimo_uplink_no_channel_reconstruction':
            x_precoded = phase_3_mu_mimo_uplink_precoding(x_precoded, x_rg_placeholder_precoded, precoding_matrices, rg, MU_MIMO_RG_populated, num_subcarriers_per_RB)
        else:
                ValueError("unsupported precoding method")

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.squeeze(x_precoded, axis=-1)
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        return x_precoded
