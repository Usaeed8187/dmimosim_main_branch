import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .p1_demo_precoding import weighted_mean_precoder


class P1DemoPrecoder(Layer):
    """Precoders for Phase 1 for demo"""

    def __init__(self,
                 resource_grid,
                 stream_management,
                 return_effective_channel=False,
                 dtype=tf.complex64,
                 **kwargs):
        super().__init__(trainable=False, dtype=dtype, **kwargs)
        assert isinstance(resource_grid, sionna.ofdm.ResourceGrid)
        assert isinstance(stream_management, sionna.mimo.StreamManagement)
        self._resource_grid = resource_grid
        self._stream_management = stream_management
        self._return_effective_channel = return_effective_channel
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

    def _compute_effective_channel(self, h, g):
        """Compute effective channel after precoding"""

        # Input dimensions:
        # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        # g: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx]

        # Transpose h to shape:
        # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
        #  ..., num_tx_ant]
        h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
        h = tf.cast(h, g.dtype)

        # Add one dummy dimension to g to be broadcastable to h:
        # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,...
        #  ..., num_streams_per_tx]
        g = tf.expand_dims(g, 1)

        # Compute post precoding channel:
        # [batch_size, num_rx, num_tx, num_ofdm, fft_size, num_rx_ant,...
        #  ..., num_streams_per_tx]
        h_eff = tf.matmul(h, g)

        # Permute dimensions to common format of channel tensors:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm, fft_size]
        h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])

        # Remove nulled subcarriers:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm, num_effective_subcarriers]
        h_eff = self._remove_nulled_scs(h_eff)

        return h_eff

    def call(self, inputs):

        ue_rank_adapt = False
        if len(inputs) == 5:
            x, h, ue_indices, ue_ranks, precoding_method = inputs
        else:
            ValueError("calling BD precoder with incorrect params")

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [num_rx, num_rx_ant, num_tx, num_tx_ant]
        num_tx, num_streams_per_tx = x.shape[1:3]
        assert num_streams_per_tx <= 2, "Invalid number of transmitted streams"

        # Transformations to bring h and x in the desired shapes

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
    
        # Transpose h:
        # [num_tx, num_streams_per_tx, num_tx_ant]
        h = np.squeeze(h)
        h_pc_desired = tf.transpose(h, [0, 2, 1])

        x_precoded, g = weighted_mean_precoder(x_precoded,
                                        h_pc_desired,
                                        return_precoding_matrix=True)

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        else:
            return x_precoded
