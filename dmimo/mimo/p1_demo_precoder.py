import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import sionna
from sionna.utils import flatten_dims
from sionna.ofdm import RemoveNulledSubcarriers

from .p1_demo_precoding import weighted_mean_precoder, wmmse_precoder

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

        # Transpose h to shape:
        # [num_rx, num_rx_ant, num_tx_ant]
        h = tf.transpose(h, [0, 2, 1])
        h = tf.cast(h, g.dtype)

        # Compute post precoding channel:
        # [num_rx, num_rx_ant, num_streams_per_tx]
        h_eff = tf.matmul(h, g)

        return h_eff

    def call(self, inputs):

        ue_rank_adapt = False
        if len(inputs) == 4:
            x, h, rx_snr_db, precoding_method = inputs
        else:
            ValueError("calling BD precoder with incorrect params")

        # x has shape
        # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        #
        # h has shape
        # [num_rx, batch_size, num_tx, num_tx_ant, num_streams_per_tx]
        num_tx, num_streams_per_tx = x.shape[1:3]
        assert num_streams_per_tx <= 2, "Invalid number of transmitted streams"

        # Transformations to bring h and x in the desired shapes

        # Transpose x:
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
        x_precoded = tf.transpose(x, [0, 1, 3, 4, 2])
        x_precoded = tf.cast(x_precoded, self._dtype)
    
        # Transpose h:
        # [num_rx, num_streams_per_tx, num_tx_ant]
        h = np.squeeze(h, axis=(1, 2))
        h_pc_desired = tf.transpose(h, [0, 2, 1])

        if precoding_method == 'baseline':
            x_precoded, g, starting_SINR, best_SINR = weighted_mean_precoder(x_precoded,
                                    h_pc_desired,
                                    rx_snr_db,
                                    num_iterations=0,
                                    return_precoding_matrix=True)
        elif precoding_method == 'weighted_mean':
            x_precoded, g, starting_SINR, best_SINR = weighted_mean_precoder(x_precoded,
                                                        h_pc_desired,
                                                        rx_snr_db,
                                                        num_iterations=3,
                                                        return_precoding_matrix=True)
        elif precoding_method == 'wmmse':
            x_precoded, g, Hg = wmmse_precoder(x_precoded,
                                                        h_pc_desired,
                                                        rx_snr_db,
                                                        num_iterations=10,
                                                        return_precoding_matrix=True)
            starting_SINR = None
            best_SINR = None
        else:
            ValueError("unsupported precoding method for phase 1 demo")

        # Transpose output to desired shape:
        # [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        x_precoded = tf.transpose(x_precoded, [0, 1, 4, 2, 3])

        if self._return_effective_channel:
            h_eff = self._compute_effective_channel(h, g)
            return x_precoded, h_eff, starting_SINR, best_SINR
        else:
            return x_precoded, None, starting_SINR, best_SINR
