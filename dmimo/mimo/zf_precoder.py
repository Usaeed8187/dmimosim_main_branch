import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .zf_precoding import sumimo_zf_precoder, sumimo_zf_precoder_modified, mumimo_zf_precoder


class ZFPrecoder(Layer):
    """ZF Precoder for MU-MIMO"""

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
        if len(inputs) == 2:
            # not using rank adaptation
            x, h = inputs
            sinr_calculation = False
        elif len(inputs) == 3:
            x, h, ue_ranks = inputs
            sinr_calculation = True
        elif len(inputs) == 4:
            # specify user Rx antennas indices and streams (rank)
            x, h, ue_indices, ue_ranks = inputs
            if ue_indices is not None and ue_ranks is not None:
                ue_rank_adapt = True
                if np.size(np.array(ue_ranks)) == 1:
                    ue_ranks = np.repeat(ue_ranks, len(ue_indices), axis=0)
            sinr_calculation = False
        else:
            ValueError("calling BD precoder with incorrect params")

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        num_tx, num_streams_per_tx = x.shape[1:3]
        num_rx, num_rx_ant = h.shape[1:3]
        assert num_streams_per_tx <= tf.size(ue_indices), "Invalid number of transmitted streams"

        # Transformations to bring h and x in the desired shapes

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
    
        # Transpose h:
        # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])

        # Gather desired channel for precoding:
        # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        # h_pc_desired = tf.gather(h_pc, tf.reshape(ue_indices, (1,-1)), axis=2, batch_dims=1)

        # Flatten dims 2,3:
        # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols, fft_size, batch_size]
        h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

        # Transpose:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_rx_ant, num_tx_ant]
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, self._dtype)

        # Rx antenna indices for MU-MIMO
        if sinr_calculation:
            if ue_rank_adapt is False:
                x_precoded, g = sumimo_zf_precoder_modified(x_precoded,
                                                h_pc_desired,
                                                ue_ranks,
                                                return_precoding_matrix=True)
        else:
            if ue_rank_adapt is False:
                x_precoded, g = sumimo_zf_precoder(x_precoded,
                                                h_pc_desired,
                                                return_precoding_matrix=True)
            else:
                # check rx_indices and rx_ranks
                num_rx_ant = [len(val) for val in ue_indices]
                total_rx_ant = np.sum(num_rx_ant)
                assert total_rx_ant == h_pc_desired.shape[4], "total number of UE antennas must match channel coefficients"
                assert all(ue_ranks <= num_rx_ant), "UE rank should not exceed number of antennas"

                x_precoded, g = mumimo_zf_precoder(x_precoded,
                                                h_pc_desired,
                                                ue_indices,
                                                ue_ranks,
                                                return_precoding_matrix=True)

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff
        else:
            return x_precoded
